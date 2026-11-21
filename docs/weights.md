# Weights and licences

Every model the pipeline loads, its licence and how to get it. The README's "Weights" gives the commands; paths in this file are relative to the repository root.

`scripts/fetch_weights.py` pins every model to a revision and verifies each published sha256; it never writes under
`third_party/`.

```bash
uv run --frozen python scripts/fetch_weights.py            # the table: source, revision, licence, sha256
uv run --frozen hf auth login                              # after accepting SAM 3D's terms on Hugging Face
uv run --frozen python scripts/fetch_weights.py --yes \
  --only sam2,moge2,sam3d,moge1_sam3d,dinov2_sam3d,cotracker3,hawor,wilor_detector --hawor-root ../hawor
```

`../hawor` is the HaWoR checkout that `scripts/setup_all.sh` cloned ("Installation"); pass yours if it is elsewhere.

`--yes` means you accept the licence of each model you fetch; several are non-commercial (see README, "Licence"). SAM 3D
Objects is gated on Hugging Face.

What each part is licensed under, and who must fetch it. `release/MODELS.md` adds the pinned revision and sha256 of
every file, and `THIRD_PARTY_LICENSES.md` the full list with the licence texts.

| part | used for | licence | commercial use | how you get it |
|---|---|---|---|---|
| this repository's code | everything | Apache-2.0 (`LICENSE`, `NOTICE`) | the code, yes; the pipeline as shipped, no (README, "Licence"; `docs/licensing.md`) | in this repository |
| SAM 2.1 Hiera-L | mask | Apache-2.0 | yes | `fetch_weights.py` |
| SAM 3D Objects | shape | SAM License | no, as the shape stage runs (nvdiffrast and kaolin, `docs/licensing.md`) | gated on Hugging Face: accept its terms, then `fetch_weights.py` |
| MoGe ViT-L (v1), inside the SAM 3D worker | shape | MIT | yes | `fetch_weights.py` |
| DINOv2 ViT-L/14 with registers, inside SAM 3D | shape | Apache-2.0 (backbone code and weights) | yes | `fetch_weights.py` |
| MoGe-2 ViT-L | intrinsics, geometry | MIT | yes | `fetch_weights.py` |
| CoTracker3 online | tracks | CC BY-NC 4.0 | no | `fetch_weights.py` |
| HaWoR | hands | CC BY-NC-ND 4.0 | no | `fetch_weights.py --hawor-root` |
| WiLoR hand detector, loaded by the HaWoR driver | hands | CC BY-NC-ND 4.0 on the stricter of two upstream statements | no | `fetch_weights.py --hawor-root` |
| Hands23 detector (opt-in prompt stage) | prompt | code MIT; the weights carry no licence text | no | `fetch_weights.py --only hands23` |
| MANO hand model | hands, retarget | MANO licence: non-commercial, no redistribution | no | register and download it yourself; never fetched or shipped |
| DO AS I DO code (`scripts/fetch_baseline.sh`, commit `824591b8`) | the baseline; the SAM 3D code that the shape stage runs | MIT for its own files; the model code inside it keeps its upstream licence (`THIRD_PARTY_LICENSES.md`) | see `THIRD_PARTY_LICENSES.md` | `git clone --recurse-submodules`; from the plain archive, `bash scripts/fetch_baseline.sh` |
| the example clip (V2D Challenge Track 3, NVIDIA) | example input | CC BY 4.0 | under CC BY 4.0, with attribution | downloaded at a pinned revision on first use; not shipped |
| the reference output and the picture at the top (`release/reference/`) | expected output | not Apache-2.0: derived from the CC BY 4.0 clip; its hand arrays follow HaWoR's CC BY-NC-ND 4.0 and MANO's licence | no | in this repository (README, "Licence") |
| the held-out unit lists (`reproduce/heldout/`) | the held-out test's unit IDs | not Apache-2.0: derived from HOI4D and DexYCB, both CC BY-NC 4.0 | no | in this repository |

**MANO** is never downloaded or shipped. Register at <https://mano.is.tue.mpg.de>, accept the licence
(<https://mano.is.tue.mpg.de/license.html>; non-commercial, no redistribution), put `MANO_RIGHT.pkl` and
`MANO_LEFT.pkl` in `FASTDO_MANO_DIR` and `FASTDO_MANO_LEFT_DIR`, and copy them into the HaWoR checkout
(`release/QUICKSTART.md` step 3).
