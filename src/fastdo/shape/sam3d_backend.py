"""SAM 3D Objects shape backend (D4): gated and optional, for the parity comparison with DO AS I DO.

SAM 3D Objects needs its own environment (the baseline's `sam3d` conda env), so this backend drives a persistent worker
process: `sam3d_worker.py` under that interpreter. The worker loads the pipeline once (69 s measured) and serves every
later clip, so the `shape` stage pays the load once per engine process rather than once per clip.
`Sam3dWorkerClient` speaks the worker's JSON-lines protocol (documented in `sam3d_worker`) and passes arrays through
npz files in a private temporary directory.

Access. The checkpoints are HF-gated. `resolve_sam3d_pipeline_config` reads them from the local Hugging Face cache only
(`local_files_only=True`), so nothing is downloaded here, at the commit `fastdo.model_pins` pins for the repository
(the one `scripts/fetch_weights.py` downloads; a download at a commit writes no `refs/main`, so an unpinned lookup
would not find it) and with the `checkpoints/*` pattern that script fetches (a snapshot holding only those files is
complete for it). When no snapshot is reachable, whether because the repo is gated or because it was never cached, it
raises `RuntimeError` starting with `SAM3D_GATED_MESSAGE`, the gated-access pattern. It never
falls back to a fabricated mesh.

Memory. SAM 3D peaks at 18.67 GiB allocated on the whisking clip. The worker applies the engine's per-process cap from
`FASTDO_GPU_MEM_GB` (default 12 GiB). On CUDA, a cap below `sam3d_worker.SAM3D_MIN_GPU_GB` (19 GiB) makes it refuse
to start, before the pipeline load, so this backend raises a clear `RuntimeError` instead of an out-of-memory error on
the first clip. A real run therefore needs `FASTDO_GPU_MEM_GB >= 19`.

Shape cache identities: `sam3d_checkpoint_identity` names the snapshot commit and the content-hash blob
of every checkpoint file (no GB-scale hashing), and `sam3d_env_identity` probes the resolved worker interpreter without
importing torch or sam3d_objects.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import select
import shutil
import subprocess
import tempfile
import weakref
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fastdo.model_pins import hf_revision, sam3d_hf_revisions
from fastdo.shape.backend import ShapeConfig, ShapeResult, register_shape_backend

__all__ = [
    "ALLOC_CONF_ENV",
    "EXPANDABLE_SEGMENTS",
    "SAM3D_GATED_MESSAGE",
    "SAM3D_PYTHONPATH_ENV",
    "SAM3D_PYTHON_ENV",
    "SAM3D_SNAPSHOT_PATTERNS",
    "WORKER_SCRIPT",
    "Sam3dBackend",
    "Sam3dWorkerClient",
    "resolve_sam3d_pipeline_config",
    "resolve_sam3d_pythonpath",
    "sam3d_alloc_conf",
    "sam3d_checkpoint_identity",
    "sam3d_env_identity",
    "sam3d_worker_env",
    "start_sam3d_worker",
]

logger = logging.getLogger(__name__)

SAM3D_GATED_MESSAGE = "SAM 3D Objects access not authorized for this account"
SAM3D_PYTHON_ENV = "FASTDO_SAM3D_PYTHON"
SAM3D_PYTHONPATH_ENV = "FASTDO_SAM3D_PYTHONPATH"
WORKER_SCRIPT = Path(__file__).with_name("sam3d_worker.py")
_PIPELINE_YAML = Path("checkpoints") / "pipeline.yaml"
# The files of the snapshot the shape stage reads: what scripts/fetch_weights.py downloads (no README or doc images).
SAM3D_SNAPSHOT_PATTERNS: tuple[str, ...] = ("checkpoints/*",)
_PINNED = object()            # resolve_sam3d_pipeline_config's default revision: the fastdo.model_pins entry
# The allocator is a property of the worker process, selected at start and part of its identity.
ALLOC_CONF_ENV = "PYTORCH_CUDA_ALLOC_CONF"
EXPANDABLE_SEGMENTS = "expandable_segments:True"


def resolve_sam3d_pipeline_config(repo_id: str, snapshot_fn: Callable[..., str] | None = None,
                                  revision: Any = _PINNED) -> Path:
    """`<local snapshot of repo_id>/checkpoints/pipeline.yaml`, from the Hugging Face cache only.

    Args:
        repo_id: The gated checkpoint repository.
        snapshot_fn: `huggingface_hub.snapshot_download`-compatible callable; always called with
            `local_files_only=True`, `revision` and `allow_patterns=SAM3D_SNAPSHOT_PATTERNS`.
        revision: The snapshot commit; by default `fastdo.model_pins.hf_revision(repo_id)` (None, i.e. the cached
            `refs/main`, for a repository the table does not pin).

    Raises:
        RuntimeError: Starting with `SAM3D_GATED_MESSAGE`, if no snapshot is reachable or it lacks the pipeline YAML.
    """
    if snapshot_fn is None:
        from huggingface_hub import snapshot_download

        snapshot_fn = snapshot_download
    if revision is _PINNED:
        revision = hf_revision(repo_id)
    try:
        snapshot = Path(snapshot_fn(repo_id=repo_id, revision=revision, allow_patterns=list(SAM3D_SNAPSHOT_PATTERNS),
                                    local_files_only=True))
    except (OSError, ValueError) as exc:  # GatedRepoError, LocalEntryNotFoundError and HTTP errors are OSErrors
        raise RuntimeError(
            f"{SAM3D_GATED_MESSAGE}: {repo_id} is HF-gated and no local snapshot at revision {revision or 'main'} is "
            f"reachable ({type(exc).__name__}: {exc}). Request access on its model page and cache it outside the "
            "engine (scripts/fetch_weights.py --yes --only sam3d)."
        ) from exc
    config = snapshot / _PIPELINE_YAML
    if not config.is_file():
        raise RuntimeError(f"{SAM3D_GATED_MESSAGE}: the gated {repo_id} snapshot at {snapshot} has no {_PIPELINE_YAML}")
    return config


def _terminate(process: subprocess.Popen[str], workdir: Path) -> None:
    if process.poll() is None:
        process.kill()
        process.wait()
    for stream in (process.stdin, process.stdout):
        if stream is not None:
            try:
                stream.close()
            except OSError:
                pass
    shutil.rmtree(workdir, ignore_errors=True)


class Sam3dWorkerClient:
    """A running `sam3d_worker.py` process; the constructor returns once the worker's pipeline has loaded.

    Args:
        command: The worker's argv.
        env: Its environment (default: inherited).
        startup_timeout_s: Longest wait for the ready line.
        request_timeout_s: Longest wait for one reply.
        spconv_kernels: The kernel mode the ready line must report ("pinned" or "tuner"); None checks nothing.

    Raises:
        RuntimeError: If the worker fails to load its pipeline, exits, times out, or runs another kernel mode.
    """

    def __init__(self, command: Sequence[str], env: Mapping[str, str] | None = None, startup_timeout_s: float = 900.0,
                 request_timeout_s: float = 900.0, spconv_kernels: str | None = None) -> None:
        self.request_timeout_s = request_timeout_s
        self._workdir = Path(tempfile.mkdtemp(prefix="fastdo_sam3d_"))
        self._next_id = 0
        self._process = subprocess.Popen(list(command), stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                         bufsize=1, env=dict(env) if env is not None else None)
        self._finalizer = weakref.finalize(self, _terminate, self._process, self._workdir)
        ready = self._read_reply(startup_timeout_s, "startup")
        if not ready.get("ready"):
            self.close()
            raise RuntimeError(f"SAM 3D worker failed to load its pipeline: {ready.get('error', ready)}")
        self.spconv_kernels: str | None = ready.get("spconv_kernels")
        if spconv_kernels is not None and self.spconv_kernels != spconv_kernels:
            self.close()
            raise RuntimeError(f"SAM 3D worker runs spconv kernels {self.spconv_kernels!r}, expected {spconv_kernels!r}")
        self.load_seconds = float(ready.get("load_seconds", 0.0))
        self.last_reply: dict[str, Any] = {}
        self.last_peak_gpu_gib: float | None = None           # max_memory_allocated of the last GATED pass (GiB)
        self.last_peak_gpu_gib_reserved: float | None = None  # max_memory_reserved, what the CUDA cap bounds
        # A wall-QC fallback's own peaks, beside the gated pass's and never merged into
        # them, so the memory gate's non-inferiority comparison (on the reserved peak) reads the arm the
        # row names. None when no fallback ran.
        self.last_fallback_peak_gpu_gib: float | None = None
        self.last_fallback_peak_gpu_gib_reserved: float | None = None
        self.last_digests: dict[str, str] | None = None       # sha256 digests when the request asked for them
        self.last_postprocess_stats: dict[str, Any] | None = None   # only when the fast path ran
        logger.info("SAM 3D worker %d ready, pipeline loaded in %.1f s", self._process.pid, self.load_seconds)

    @property
    def pid(self) -> int:
        return self._process.pid

    def reconstruct(self, image: np.ndarray, mask: np.ndarray, params: Mapping[str, Any]) -> ShapeResult:
        """One geometry-only reconstruction by the worker; `params` are `reconstruct_geometry`'s keyword arguments,
        plus the optional `digest`. Sets `last_reply`, `last_peak_gpu_gib`, `last_peak_gpu_gib_reserved` (None off
        CUDA), their `last_fallback_*` counterparts and `last_digests`.

        The result's `phase_seconds` is the gated pass alone, so it sums to the gated worker seconds; a wall-QC
        fallback's phases are returned separately as `fallback_phase_seconds` (None when none ran). The peaks are
        split the same way and for the same reason: `last_peak_gpu_gib_reserved` is the gated pass's,
        which is what the memory gate reads."""
        request_id, self._next_id = self._next_id, self._next_id + 1
        request, result = self._workdir / f"request_{request_id}.npz", self._workdir / f"result_{request_id}.npz"
        try:
            with request.open("wb") as f:
                np.savez(f, image=np.asarray(image, dtype=np.uint8), mask=np.asarray(mask, dtype=bool))
            self._send({"cmd": "reconstruct", "id": request_id, "input": str(request), "output": str(result),
                        "params": dict(params)})
            reply = self._read_reply(self.request_timeout_s, f"request {request_id}")
            if reply.get("id") != request_id:
                raise RuntimeError(f"SAM 3D worker answered request {reply.get('id')!r}, expected {request_id}")
            self.last_reply = reply
            self.last_peak_gpu_gib = reply.get("peak_gpu_gib")
            self.last_peak_gpu_gib_reserved = reply.get("peak_gpu_gib_reserved")
            self.last_fallback_peak_gpu_gib = reply.get("fallback_peak_gpu_gib")
            self.last_fallback_peak_gpu_gib_reserved = reply.get("fallback_peak_gpu_gib_reserved")
            self.last_digests = reply.get("digests")
            self.last_postprocess_stats = reply.get("postprocess_stats")
            if not reply.get("ok"):
                raise RuntimeError(f"SAM 3D worker failed on request {request_id}: {reply.get('error')}")
            with np.load(result, allow_pickle=False) as data:
                vertices = np.asarray(data["vertices"], dtype=np.float64)
                faces = np.asarray(data["faces"], dtype=np.int64)
        finally:
            request.unlink(missing_ok=True)
            result.unlink(missing_ok=True)
        phases = {name: float(seconds) for name, seconds in reply.get("phase_seconds", {}).items()}
        # The reply keeps the gated pass and a QC fallback apart, and so does the result. Merging
        # them here (even under a `fallback_` prefix) would stop `phase_seconds` summing to the gated worker seconds,
        # which is what the timing gate and t_eff are computed from.
        fallback = reply.get("fallback_phase_seconds")
        fallback_phases = {name: float(seconds) for name, seconds in fallback.items()} if fallback else None
        return ShapeResult(vertices, faces, None, phases, fallback_phases)

    def close(self, timeout_s: float = 60.0) -> int | None:
        """Asks the worker to exit (killing it after `timeout_s`); returns its exit code."""
        if self._process.poll() is None:
            try:
                self._send({"cmd": "shutdown"})
                self._process.wait(timeout=timeout_s)
            except (OSError, subprocess.TimeoutExpired):
                logger.warning("SAM 3D worker %d did not exit cleanly; killing it", self._process.pid)
        self._finalizer()
        return self._process.returncode

    def _send(self, message: dict[str, Any]) -> None:
        stdin = self._process.stdin
        assert stdin is not None
        stdin.write(json.dumps(message) + "\n")
        stdin.flush()

    def _read_reply(self, timeout_s: float, what: str) -> dict[str, Any]:
        stdout = self._process.stdout
        assert stdout is not None
        readable, _, _ = select.select([stdout], [], [], timeout_s)
        if not readable:
            self._finalizer()
            raise RuntimeError(f"SAM 3D worker timed out after {timeout_s:.0f} s during {what}; killed")
        line = stdout.readline()
        if not line:
            code = self._process.wait()
            self._finalizer()
            raise RuntimeError(f"SAM 3D worker exited with code {code} during {what}; its traceback is on stderr")
        reply: dict[str, Any] = json.loads(line)
        return reply


def resolve_sam3d_pythonpath(cfg: ShapeConfig) -> tuple[str, ...]:
    """The SAM 3D worker's extra PYTHONPATH entries: `cfg.sam3d_pythonpath`, or, when it is None, the
    os.pathsep-separated `FASTDO_SAM3D_PYTHONPATH` (unset or empty: none; empty segments are skipped)."""
    if cfg.sam3d_pythonpath is not None:
        return cfg.sam3d_pythonpath
    return tuple(entry for entry in os.environ.get(SAM3D_PYTHONPATH_ENV, "").split(os.pathsep) if entry)


def sam3d_alloc_conf(cfg: ShapeConfig) -> str | None:
    """The `PYTORCH_CUDA_ALLOC_CONF` the worker starts with, or None to leave the allocator alone.

    The EVAL's memory arms need `expandable_segments:True`: with the default allocator, Euler 8+8 fragments after a
    larger decode and OOMs at 20.29 GiB reserved in one request order, while the same order under expandable segments
    peaks at 18.42-18.46 GiB with byte-identical outputs.
    """
    return EXPANDABLE_SEGMENTS if cfg.sam3d_expandable_segments else None


def sam3d_worker_env(extra: Sequence[str], environ: Mapping[str, str] | None = None,
                     alloc_conf: str | None = None) -> dict[str, str] | None:
    """A copy of `environ` (default `os.environ`) with `extra` prepended to PYTHONPATH, in order, ahead of any
    existing entries, and `alloc_conf` set as `PYTORCH_CUDA_ALLOC_CONF` when it is given; None (the worker inherits
    the environment unchanged, as it does today) when there are no entries and no allocator setting."""
    if not extra and alloc_conf is None:
        return None
    env = dict(os.environ if environ is None else environ)
    if extra:
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join([*extra, existing] if existing else extra)
    if alloc_conf is not None:
        env[ALLOC_CONF_ENV] = alloc_conf
    return env


def sam3d_checkpoint_identity(repo_id: str, snapshot_fn: Callable[..., str] | None = None) -> str:
    """`"sam3d:<snapshot commit>:<name>=<blob>,..."` over the snapshot's `checkpoints/*`, sorted by name; Hugging Face
    blob names are content hashes.

    Raises:
        RuntimeError: As `resolve_sam3d_pipeline_config`.
        ValueError: If a checkpoint entry is not a symlink (not a Hugging Face cache layout).
    """
    checkpoints = resolve_sam3d_pipeline_config(repo_id, snapshot_fn).parent
    entries = []
    for path in sorted(checkpoints.iterdir(), key=lambda p: p.name):
        if not path.is_symlink():
            raise ValueError(f"{path} is not a symlink into the Hugging Face blob store; the shape cache cannot "
                             "identify the checkpoint without hashing it")
        entries.append(f"{path.name}={os.path.basename(os.readlink(path))}")
    return f"sam3d:{checkpoints.parent.name}:{','.join(entries)}"


# Runs under the SAM 3D interpreter: stdlib only, never imports torch or sam3d_objects.
_ENV_PROBE = r"""
import hashlib, importlib.metadata as md, importlib.util, json, os, sys
spec = importlib.util.find_spec("sam3d_objects")
if spec is None or not spec.submodule_search_locations:
    sys.exit("sam3d_objects is not importable")
root = list(spec.submodule_search_locations)[0]
h = hashlib.sha256()
for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d != "__pycache__")
    for name in sorted(filenames):
        if name.endswith((".pyc", ".pyo")):
            continue
        path = os.path.join(dirpath, name)
        with open(path, "rb") as f:
            data = f.read()
        rel = os.path.relpath(path, root).encode()
        h.update(rel + b"\0" + len(data).to_bytes(8, "little") + data)
dists = {}
for dist in md.distributions():
    name = dist.metadata["Name"]
    if not name:
        continue
    record = dist.read_text("RECORD")
    dists.setdefault(name, [dist.version, hashlib.sha256(record.encode()).hexdigest() if record else None])  # first wins, as import
keys = ("SPCONV_ALGO", "ATTN_BACKEND", "SPARSE_ATTN_BACKEND", "SPARSE_BACKEND", "CUBLAS_WORKSPACE_CONFIG")
print(json.dumps({"executable": os.path.realpath(sys.executable), "version": sys.version, "sam3d_objects": root,
                  "sam3d_objects_sha256": h.hexdigest(), "distributions": dict(sorted(dists.items())),
                  "env": {k: os.environ.get(k) for k in keys}, "sys_path": sys.path,
                  "torch_imported": "torch" in sys.modules}))
"""


def _gpu_names() -> list[str] | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                             text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None                        # no driver tool on this host: recorded as null
    return [line.strip() for line in out.splitlines() if line.strip()]


def sam3d_env_identity(cfg: ShapeConfig, timeout_s: float = 120.0) -> str:
    """Canonical JSON identifying the SAM 3D worker environment (Check H1): the resolved interpreter's realpath (and
    sha256 when it is a script such as a PYTHONPATH wrapper), the resolved PYTHONPATH entries, and a probe run under
    that interpreter with the worker's environment: the sam3d_objects sources' sha256, every installed distribution's
    version and RECORD sha256, the kernel-selecting environment variables and `sys.path`; plus the GPU names.

    The allocator is recorded as `alloc_conf` when `cfg.sam3d_expandable_segments` pins it, and
    otherwise when the ambient environment exports a non-empty `PYTORCH_CUDA_ALLOC_CONF`, which is what the worker
    and this probe both inherit: a record that could not see the allocator it ran under would silently claim the
    default one and corrupt the allocator and default-allocator memory comparisons. It is absent when neither
    sets one, so a default request's identity and cache key are unchanged. The probe's own `keys` tuple
    deliberately omits the allocator for that same reason: naming it there would emit a null entry in *every*
    identity and move every default key.

    Raises:
        ValueError: If no interpreter is configured or it cannot be run, or the probe fails (for example
            sam3d_objects is not importable).
    """
    python = cfg.sam3d_python or os.environ.get(SAM3D_PYTHON_ENV)
    if not python or not os.path.isfile(python):
        raise ValueError(f"the shape cache cannot identify the SAM 3D interpreter {python!r} (ShapeConfig.sam3d_python "
                         f"or {SAM3D_PYTHON_ENV})")
    extra = resolve_sam3d_pythonpath(cfg)
    alloc_conf = sam3d_alloc_conf(cfg)
    with open(python, "rb") as f:
        script = f.read(2) == b"#!"
    try:
        proc = subprocess.run([python, "-c", _ENV_PROBE], capture_output=True, text=True, timeout=timeout_s,
                              env=sam3d_worker_env(extra, alloc_conf=alloc_conf), check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError(f"the SAM 3D interpreter {python!r} could not run the environment probe: {exc}") from exc
    if proc.returncode != 0:
        raise ValueError(f"the SAM 3D environment probe failed under {python!r}: {proc.stderr.strip()[-500:]}")
    try:
        probe = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"the SAM 3D environment probe printed no JSON: {proc.stdout[-200:]!r}") from exc
    identity: dict[str, Any] = {"interpreter": os.path.realpath(python),
                                "interpreter_sha256": (hashlib.sha256(Path(python).read_bytes()).hexdigest()
                                                       if script else None),
                                "pythonpath": list(extra), "probe": probe, "gpu": _gpu_names()}
    recorded = alloc_conf or os.environ.get(ALLOC_CONF_ENV) or None
    if recorded is not None:        # absent when neither the config nor the environment sets one, so an unpinned
        identity["alloc_conf"] = recorded       # worker's identity, and every default cache key, is unchanged
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def start_sam3d_worker(cfg: ShapeConfig, snapshot_fn: Callable[..., str] | None = None,
                       startup_timeout_s: float = 900.0) -> Sam3dWorkerClient:
    """Resolves the gated checkpoints (local cache only), then starts and waits for a `sam3d_worker.py` process.

    The worker inherits this process's environment, with `resolve_sam3d_pythonpath(cfg)` prepended to its PYTHONPATH
    when there are any entries (`sam3d_worker_env`); only those entries are logged, never the environment.

    Raises:
        RuntimeError: If access is not authorized (`SAM3D_GATED_MESSAGE`), no interpreter is configured, or the
            worker fails to start.
    """
    config = resolve_sam3d_pipeline_config(cfg.sam3d_repo, snapshot_fn)
    python = cfg.sam3d_python or os.environ.get(SAM3D_PYTHON_ENV)
    if not python:
        raise RuntimeError(f"no interpreter for the SAM 3D worker: set ShapeConfig.sam3d_python or {SAM3D_PYTHON_ENV} "
                           "to the SAM 3D environment's python")
    extra = resolve_sam3d_pythonpath(cfg)
    alloc_conf = sam3d_alloc_conf(cfg)                            # the allocator, at process start
    kernels = "pinned" if cfg.spconv_kernel_pin else "tuner"      # sam3d_spconv_pin in the worker
    command = [python, str(WORKER_SCRIPT), "--config", str(config), "--device", cfg.device, "--spconv-kernels", kernels]
    for repo, rev in sam3d_hf_revisions().items():              # the pipeline YAML's own HF loaders (MoGe v1)
        command += ["--hf-revision", f"{repo}={rev}"]
    logger.info("starting the SAM 3D worker: %s", " ".join(command))
    if extra:
        logger.info("SAM 3D worker PYTHONPATH prepends %s", list(extra))
    if alloc_conf is not None:
        logger.info("SAM 3D worker %s=%s", ALLOC_CONF_ENV, alloc_conf)
    return Sam3dWorkerClient(command, env=sam3d_worker_env(extra, alloc_conf=alloc_conf),
                             startup_timeout_s=startup_timeout_s, spconv_kernels=kernels)


@register_shape_backend("sam3d")
class Sam3dBackend:
    """The `sam3d` shape backend: one persistent worker per backend instance, started on first use.

    Args:
        cfg: Shape settings (step counts, solver, seed, mesh post-process, interpreter).
        client: An already-started `Sam3dWorkerClient`-compatible object (tests inject fakes).
        loader: Zero-argument callable returning the client, called once instead of `start_sam3d_worker(cfg)`.
        snapshot_fn: `huggingface_hub.snapshot_download`-compatible callable for `cache_identity` (tests).
        env_probe: Callable replacing `sam3d_env_identity` in `cache_env_identity` (tests).

    Raises:
        RuntimeError: If `client`, or the one `loader` returns (closed first), reports a `spconv_kernels` mode other
            than `cfg.spconv_kernel_pin`'s: shape.json and the shape cache key take the mode from `cfg`.
    """

    def __init__(self, cfg: ShapeConfig | None = None, client: Any = None, loader: Callable[[], Any] | None = None,
                 snapshot_fn: Callable[..., str] | None = None,
                 env_probe: Callable[[ShapeConfig], str] | None = None) -> None:
        self.cfg = cfg if cfg is not None else ShapeConfig(backend="sam3d")
        self._client = self._checked_kernels(client) if client is not None else None
        self._loader = loader
        self._snapshot_fn = snapshot_fn
        self._env_probe = env_probe

    @property
    def loaded(self) -> bool:
        return self._client is not None

    @property
    def sampling_params(self) -> dict[str, Any]:
        """The geometry-only request parameters `reconstruct_geometry` takes."""
        return {"seed": self.cfg.seed, "sparse_structure_steps": self.cfg.sparse_structure_steps,
                "structured_latent_steps": self.cfg.structured_latent_steps, "solver": self.cfg.solver,
                "mesh_postprocess": self.cfg.mesh_postprocess, "deterministic": self.cfg.deterministic,
                # Per request, so one worker process can serve every EVAL arm.
                "mesh_postprocess_impl": self.cfg.mesh_postprocess_impl,
                "fast_decimator_min_raw_faces": self.cfg.fast_decimator_min_raw_faces,
                "min_component_area_frac": self.cfg.min_component_area_frac,
                "wall_qc": self.cfg.wall_qc, "wall_qc_max_removed": self.cfg.wall_qc_max_removed,
                "wall_qc_min_area_ratio": self.cfg.wall_qc_min_area_ratio,
                # Only when set, so a default request is byte-identical on the wire.
                **({"sam3d_cuda_graph_ss": True} if self.cfg.sam3d_cuda_graph_ss else {}),
                # Likewise.
                **({"sam3d_cuda_graph_slat": True} if self.cfg.sam3d_cuda_graph_slat else {}),
                # The sampler options, likewise only when set.
                **({"sparse_structure_shortcut": True} if self.cfg.sparse_structure_shortcut else {}),
                **({"structured_latent_solver": self.cfg.structured_latent_solver}
                   if self.cfg.structured_latent_solver is not None else {}),
                # The sparse-structure flow's solver, likewise only when set.
                **({"sparse_structure_solver": self.cfg.sparse_structure_solver}
                   if self.cfg.sparse_structure_solver is not None else {})}

    @property
    def last_postprocess_stats(self) -> dict[str, Any] | None:
        """The worker's `postprocess_stats` for the last request (None unless the fast post-process ran)."""
        client = self._client
        return None if client is None else getattr(client, "last_postprocess_stats", None)

    def cache_identity(self) -> str:
        """The shape cache's checkpoint part (`sam3d_checkpoint_identity`); loads nothing."""
        return sam3d_checkpoint_identity(self.cfg.sam3d_repo, self._snapshot_fn)

    def cache_env_identity(self) -> str:
        """The shape cache's environment part (`sam3d_env_identity`); starts no worker."""
        return (self._env_probe or sam3d_env_identity)(self.cfg)

    def load(self) -> Any:
        """The worker client, started on first call and reused."""
        if self._client is None:
            client = self._loader() if self._loader is not None else start_sam3d_worker(self.cfg)
            try:
                self._client = self._checked_kernels(client)
            except RuntimeError:
                client.close()
                raise
        return self._client

    def _checked_kernels(self, client: Any) -> Any:
        mode, expected = getattr(client, "spconv_kernels", None), "pinned" if self.cfg.spconv_kernel_pin else "tuner"
        if isinstance(mode, str) and mode != expected:              # None or a fake without the field: unreported
            raise RuntimeError(f"SAM 3D worker runs spconv kernels {mode!r}, but the config expects {expected!r}")
        return client

    def reconstruct(self, image: np.ndarray, mask: np.ndarray) -> ShapeResult:
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("mask is empty: the shape backend needs at least one object pixel")
        result: ShapeResult = self.load().reconstruct(np.asarray(image), mask, self.sampling_params)
        return result

    def close(self) -> int | None:
        """Stops the worker, if one was started; returns its exit code."""
        client, self._client = self._client, None
        return client.close() if client is not None else None
