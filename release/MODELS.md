# External models of the reconstruction pipeline

The external files pinned for the released configurations of the reconstruction (stages decode to qc, plus the
opt-in prompt stage), with where each comes from, its licence, and the sha256 to verify it against: the digest of
the file the authors' runs had in place. The list includes files that only an opt-in stage or flag reads, files fetched by hand,
and one zero-byte placeholder. A digest pins a file; it does not show that a given run opened it. **No weights are shipped.**
`models.json` holds the same data in machine-readable form, and this page is generated from it
(`release/render_models_md.py`); `tests/release/test_release_manifest.py` keeps the two in step and checks them
against `scripts/fetch_weights.py`.

How to fetch: `uv run --frozen python scripts/fetch_weights.py` prints the download table; `--yes --only <keys>`
downloads the listed files and compares each with its SHA-256 (QUICKSTART step 3). Two things stay outside that
automatic check. MANO is a manual download that the script neither fetches nor hashes. A code directory that was
already present without the script's pin file (for example a tree that torch.hub cached) is left as it is and
reported UNVERIFIED, which is a warning and not a failure. Four model loaders (SAM 2.1, SAM 3D Objects, MoGe v1 and MoGe-2) fetch and load at the revisions of
`src/fastdo/model_pins.py`. For HaWoR and the WiLoR detector, the hand worker opens local files: verify them against
the source revisions and SHA-256 values listed below. MANO is never downloaded for you.

Path variables: `$HF_HOME` (default `~/.cache/huggingface`), `$TORCH_HOME` (default `~/.cache/torch`), and the
`FASTDO_*` variables of README.md, "Host paths".

**Outputs are bound too.** What the pipeline writes (meshes, poses, hand parameters, robot trajectories) is produced
by these models, so their licences (non-commercial, NoDerivatives, MANO's) also govern what you may do with the
outputs. Whether an output is a derivative work is an open question for several of them (THIRD_PARTY_LICENSES.md).

This states what the licence texts say as we read them. It is not legal advice.

## Summary

| key | model | stage | env | licence | commercial use | flags |
|---|---|---|---|---|---|---|
| `sam2` | SAM 2.1 Hiera-L | mask, prompt (opt-in) | uv (main) | Apache-2.0 | yes | - |
| `sam3d` | SAM 3D Objects | shape | sam3d (conda) | SAM License (custom) | **no** | shape stage non-commercial as run: nvdiffrast 0.4.0 (NVIDIA Source Code License, non-commercial, cl. 3.3) runs in its process through fast_postprocess.py -> utils3d RastContext (hole filling, on by default); kaolin 0.18.0, imported by SAM 3D's flexicubes on a static reading, loads its NVIDIA-NSCL (non-commercial) kaolin/non_commercial package; the model licence itself (SAM License) permits commercial use on our reading, with its acknowledgement and trade-control terms; read it; gated download; the checkpoint sha256s equal Hugging Face's LFS ids; the yaml digests are of our copy; ss_encoder.safetensors is empty upstream too (size 0, the git empty blob, in the HF tree at this revision) |
| `moge1_sam3d` | MoGe ViT-L (v1), the SAM 3D worker's pointmap model | shape | sam3d (conda) | MIT (MoGe repository licence) | yes | - |
| `dinov2_sam3d` | DINOv2 ViT-L/14 with registers, SAM 3D's image conditioner | shape | sam3d (conda) | Apache-2.0 (backbone code and weights) | yes | the hub code at that commit (hubconf.py) also imports the Cell-DINO modules when torch.hub loads it; their files state CC BY 4.0 (LICENSE_CELL_DINO_CODE) and, in dinov2/hub/cell_dino/backbones.py, CC BY-NC. SAM 3D calls only dinov2_vitl14_reg and loads no Cell-DINO weights (THIRD_PARTY_LICENSES.md M14); unversioned weights URL: the sha256 is of our copy |
| `moge2` | MoGe-2 ViT-L | intrinsics, geometry | uv (main) | MIT | yes | - |
| `cotracker3` | CoTracker3 online | tracks | uv (main) | CC BY-NC 4.0 | **no** | non-commercial; the engine loads the torch.hub code at commit 82e02e80 by default, the commit the paper's runs loaded (their cached main equals it file for file); FASTDO_COTRACKER_HUB_REF overrides it (THIRD_PARTY_LICENSES.md M4) |
| `hawor` | HaWoR | hands | hawor (conda) | CC BY-NC-ND 4.0 | **no** | non-commercial; no derivatives (ND); imports smplx (MPI non-commercial licence) |
| `wilor_detector` | WiLoR hand detector (YOLO), loaded by the HaWoR driver | hands | hawor (conda) | CC BY-NC-ND 4.0 (stricter of two conflicting upstream statements; the other is CC BY-NC 2.0) | **no** | non-commercial; unclear: conflicting upstream licence statements; loaded in-process by ultralytics (AGPL-3.0) |
| `hands23` | Hands23 hand-object detector (opt-in internet-video prompt stage) | prompt | hands23 (venv) | code MIT; weights carry no licence text; training data CC BY-NC 4.0 | **no** | unclear: weights have no licence text; treat as non-commercial (training data CC BY-NC); unversioned download URL |
| `mano` | MANO hand model (right and left) | hands, retarget | hawor (conda); uv (retarget) | MANO licence (non-commercial; no redistribution in any form; no training of commercial models) | **no** | non-commercial; no redistribution; registration required |

**As shipped, the pipeline is for non-commercial research only**: CoTracker3, HaWoR, the WiLoR detector, MANO and the Hands23 weights are non-commercial or carry no licence, smplx (inside HaWoR) is MPI non-commercial, and the shape stage runs NVIDIA's non-commercial nvdiffrast (and, on a static reading, kaolin's non-commercial package) in its process.

## Code licence flags in the worker environments

From the package metadata of the environments in `release/env/` (scan of `*.dist-info/METADATA`) and the licence files of kaolin and nvdiffrast. None of these packages is included in this repository. The usage column says, for each, whether the pipeline imports it at run time or the environment only installs it. Original HandReel portions of the files that run in the same process carry Apache-2.0. Adapted portions retain the upstream terms identified in THIRD_PARTY_LICENSES.md (for example the SAM-adapted parts of `src/fastdo/shape/fast_postprocess.py`). If you distribute a combined work, the copyleft terms of the packages it contains apply to it.

| package | licence | env | how the pipeline uses it | notes |
|---|---|---|---|---|
| nvdiffrast 0.4.0 | NVIDIA Source Code License (non-commercial, cl. 3.3) | sam3d | in-process: src/fastdo/shape/fast_postprocess.py:218 builds utils3d's RastContext (nvdiffrast) for hole filling, on by default; SAM 3D's postprocessing_utils imports it lazily too | the shape stage is non-commercial as run |
| kaolin 0.18.0 | Apache-2.0, except kaolin/non_commercial/ under NVIDIA LICENSE.NSCL (non-commercial) | sam3d | in-process on a static reading: SAM 3D's representations/mesh/flexicubes imports kaolin, whose __init__ imports non_commercial unconditionally; not verified at run time | the shape stage is non-commercial as run |
| pymeshfix 0.17.0 | GPL-3.0 | sam3d | in-process: imported by src/fastdo/shape/fast_postprocess.py and by SAM 3D's own postprocess | original HandReel portions Apache-2.0, adapted portions under the upstream terms of THIRD_PARTY_LICENSES.md; imported at run time, not included; its copyleft terms apply to any combined work you distribute |
| igraph 0.11.8 (formerly python-igraph) | GPL-2.0-or-later | sam3d | in-process: src/fastdo/shape/fast_postprocess.py | original HandReel portions Apache-2.0, adapted portions under the upstream terms of THIRD_PARTY_LICENSES.md; imported at run time, not included; its copyleft terms apply to any combined work you distribute |
| plyfile 1.1.3 | GPL-3.0-or-later | sam3d, hawor | in-process in the SAM 3D worker on a static reading: sam3d_objects ...representations/__init__.py imports the gaussian representation, which imports plyfile; not verified at run time | original HandReel portions Apache-2.0, adapted portions under the upstream terms of THIRD_PARTY_LICENSES.md; imported at run time, not included; its copyleft terms apply to any combined work you distribute |
| ultralytics 8.4.36 (+ ultralytics-thop 2.0.18) | AGPL-3.0 | hawor | in-process: run/pipeline/_hawor_*.py load the WiLoR detector with it | original HandReel portions Apache-2.0, adapted portions under the upstream terms of THIRD_PARTY_LICENSES.md; imported at run time, not included; its copyleft terms apply to any combined work you distribute |
| evo 1.35.2 | GPL-3.0 | hawor | installed; imported only by HaWoR's lib/pipeline/masked_droid_slam.py (SLAM), which the hands driver does not import on a static reading | installed only; listed in THIRD_PARTY_LICENSES.md |
| PyQt5 5.15.11 | GPL-3.0 | hawor | installed (dependency of the env); not imported by the pipeline on a static reading | installed only; listed in THIRD_PARTY_LICENSES.md |
| easydict, dominate, autocommand, PyQt5-Qt5 | LGPL-3.0 | sam3d, hawor | installed; LGPL (library copyleft) | note only |

## Per model

### `sam2`: SAM 2.1 Hiera-L

- **Version:** HF revision 665f8e2ad61cf5f53d65644ff27c8ee525124610
- **Code:** sam-2 git 2b90b9f5ceec907a1c18123530e92e794ad901a4 (uv.lock)
- **Source:** <https://huggingface.co/facebook/sam2.1-hiera-large>
- **Licence:** Apache-2.0 (<https://github.com/facebookresearch/sam2/blob/main/LICENSE>)
- **Fetch:** scripts/fetch_weights.py --yes --only sam2

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `sam2.1_hiera_large.pt` | 898,083,611 | `2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318` | `$HF_HOME/hub/models--facebook--sam2.1-hiera-large/snapshots/665f8e2ad61cf5f53d65644ff27c8ee525124610/sam2.1_hiera_large.pt` |

### `sam3d`: SAM 3D Objects

- **Version:** HF revision 2e73555018d2741ccd486e56c24fac41155a1dc6
- **Code:** sam-3d-objects fork @ 875b010016f4c86936985b74ffcd7230e6979d02 (inside the DO AS I DO submodule)
- **Source:** <https://huggingface.co/facebook/sam-3d-objects>
- **Licence:** SAM License (custom) (<https://huggingface.co/facebook/sam-3d-objects/blob/main/LICENSE>) **GATED**
- **Fetch:** accept the terms on the HF page, `hf auth login`, then scripts/fetch_weights.py --yes --only sam3d (checkpoints/* at this revision; the shape stage reads exactly that snapshot)

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `checkpoints/pipeline.yaml` | 3,548 | `53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/pipeline.yaml` |
| `checkpoints/slat_decoder_gs.ckpt` | 171,476,155 | `f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_decoder_gs.ckpt` |
| `checkpoints/slat_decoder_gs.yaml` | 576 | `53f054e02a0c185f0a6885d30bb6ca0ec92efe0a98148688e7f05f7c26afe670` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_decoder_gs.yaml` |
| `checkpoints/slat_decoder_gs_4.ckpt` | 170,269,801 | `731a0eceaa47945b52aa27f650d695b2aea9cc70945751e5609e5cb5b49f0186` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_decoder_gs_4.ckpt` |
| `checkpoints/slat_decoder_gs_4.yaml` | 575 | `3d1dfd4c56cdac56f30e0cf5eb310d4df973fe25c60ac0a462c86ca4e3bf8b48` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_decoder_gs_4.yaml` |
| `checkpoints/slat_decoder_mesh.ckpt` | 363,726,862 | `85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_decoder_mesh.ckpt` |
| `checkpoints/slat_decoder_mesh.pt` | 363,728,714 | `93333fcd57a3e36ded0b3bca6969e05ce2b35142029dadab514f41df46d2f985` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_decoder_mesh.pt` |
| `checkpoints/slat_decoder_mesh.yaml` | 300 | `8f46952764aa985c50109a56f0b4f07625cdb06457aaded74957d26f3520c69e` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_decoder_mesh.yaml` |
| `checkpoints/slat_encoder.ckpt` | 173,263,986 | `6485623145535f42c8afa4cbb68ab9953e54e2f0c1cb1eaf95dcb41051e10181` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_encoder.ckpt` |
| `checkpoints/slat_encoder.yaml` | 268 | `ca80da7f1395a2acef9c6cea30ec3ccc89321ef890c644860f316a4ef1e8ebdb` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_encoder.yaml` |
| `checkpoints/slat_generator.ckpt` | 4,906,537,684 | `91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_generator.ckpt` |
| `checkpoints/slat_generator.yaml` | 1,986 | `53029fadff6fe34a0344381a16d64d65d0567d603484952712e8969319559c4e` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/slat_generator.yaml` |
| `checkpoints/ss_decoder.ckpt` | 147,609,242 | `6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/ss_decoder.ckpt` |
| `checkpoints/ss_decoder.yaml` | 244 | `baacff269b664f84f7aa1896ebd66128065f6568cdb88d419d5c3d2ccb4193ae` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/ss_decoder.yaml` |
| `checkpoints/ss_encoder.ckpt` | 119,085,402 | `dcc47810ac568b11fe6e4821ea1c8d6b960dfbda3e5f68e94c19f44b3bf9e83b` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/ss_encoder.ckpt` |
| `checkpoints/ss_encoder.safetensors` | 0 | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/ss_encoder.safetensors` |
| `checkpoints/ss_encoder.yaml` | 231 | `d3df5e18b2cbe87697d319adce0fe7e4fb6266809ad9d2f8c502761739bc4a06` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/ss_encoder.yaml` |
| `checkpoints/ss_generator.ckpt` | 6,690,136,964 | `225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/ss_generator.ckpt` |
| `checkpoints/ss_generator.yaml` | 5,076 | `3c265448bca7c057f94e3ef56adea3a895a10bcd9f15f992a41dd03fa35412cd` | `$HF_HOME/hub/models--facebook--sam-3d-objects/snapshots/2e73555018d2741ccd486e56c24fac41155a1dc6/checkpoints/ss_generator.yaml` |

### `moge1_sam3d`: MoGe ViT-L (v1), the SAM 3D worker's pointmap model

- **Version:** HF revision ad326bfb61facd6c52b5a825bc1e34d7c97d9672
- **Code:** moge git a8c37341bc0325ca99b9d57981cc3bb2bd3e255b (sam3d env)
- **Source:** <https://huggingface.co/Ruicheng/moge-vitl>
- **Licence:** MIT (MoGe repository licence) (<https://github.com/microsoft/MoGe/blob/main/LICENSE>)
- **Fetch:** scripts/fetch_weights.py --yes --only moge1_sam3d; the SAM 3D worker reads it at this revision (--hf-revision)

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `model.pt` | 1,256,823,446 | `da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f` | `$HF_HOME/hub/models--Ruicheng--moge-vitl/snapshots/ad326bfb61facd6c52b5a825bc1e34d7c97d9672/model.pt` |

### `dinov2_sam3d`: DINOv2 ViT-L/14 with registers, SAM 3D's image conditioner

- **Version:** dinov2_vitl14_reg4_pretrain.pth (unversioned URL); code torch.hub facebookresearch/dinov2 @ 7764ea0f912e53c92e82eb78a2a1631e92725fc8
- **Code:** torch.hub facebookresearch/dinov2 @ 7764ea0f912e53c92e82eb78a2a1631e92725fc8 (equal file for file to the main our runs cached), unpacked by fetch_weights.py into $TORCH_HOME/hub/facebookresearch_dinov2_main, which SAM 3D's unpinned torch.hub.load reads
- **Source:** <https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_reg4_pretrain.pth>
- **Licence:** Apache-2.0 (backbone code and weights) (<https://github.com/facebookresearch/dinov2/blob/main/LICENSE>)
- **Fetch:** scripts/fetch_weights.py --yes --only dinov2_sam3d (checkpoint and code; an existing code directory is kept)

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `dinov2_vitl14_reg4_pretrain.pth` | 1,217,607,321 | `36e4deffbaef061a2576705b0c36f93621e2ae20bf6274694821b0b492551b51` | `$TORCH_HOME/hub/checkpoints/dinov2_vitl14_reg4_pretrain.pth` |

### `moge2`: MoGe-2 ViT-L

- **Version:** HF revision 39c4d5e957afe587e04eec59dc2bcc3be5ecd968
- **Code:** moge git b942f00bdc2a2a23ebb474fbe034d487e6dcceec (pyproject.toml)
- **Source:** <https://huggingface.co/Ruicheng/moge-2-vitl>
- **Licence:** MIT (<https://github.com/microsoft/MoGe/blob/main/LICENSE>)
- **Fetch:** scripts/fetch_weights.py --yes --only moge2

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `model.pt` | 1,305,030,700 | `3eefd4abb2102f38f12b2d1992e5ff15e4923e5431c67dd494afe157e0111cd5` | `$HF_HOME/hub/models--Ruicheng--moge-2-vitl/snapshots/39c4d5e957afe587e04eec59dc2bcc3be5ecd968/model.pt` |

### `cotracker3`: CoTracker3 online

- **Version:** weights HF facebook/cotracker3 revision bf55ea50d4390e1820a267f131cd6587240fb2c5; code torch.hub co-tracker @ 82e02e8029753ad4ef13cf06be7f4fc5facdda4d
- **Code:** torch.hub facebookresearch/co-tracker @ 82e02e8029753ad4ef13cf06be7f4fc5facdda4d (set FASTDO_COTRACKER_HUB_REF)
- **Source:** <https://huggingface.co/facebook/cotracker3>
- **Licence:** CC BY-NC 4.0 (<https://github.com/facebookresearch/co-tracker/blob/main/LICENSE.md>)
- **Fetch:** scripts/fetch_weights.py --yes --only cotracker3

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `scaled_online.pth` | 101,695,610 | `205d34789f19699d64b22cf93f9b697f15f28d4025240e31532e504109837218` | `$TORCH_HOME/hub/checkpoints/scaled_online.pth` |

### `hawor`: HaWoR

- **Version:** HF ThunderVVV/HaWoR revision da6335f47f9806308992d5ae1002a4cc5f7252c2; code malik-group/HaWoR fork @ 2c3fa0c06e54a1b192c1ff692eaa89d526cd6317
- **Code:** https://github.com/malik-group/HaWoR @ 2c3fa0c06e54a1b192c1ff692eaa89d526cd6317 (the fork and commit DO AS I DO pins; clone it outside third_party/ in QUICKSTART step 2; obtain the weights in step 3)
- **Source:** <https://huggingface.co/ThunderVVV/HaWoR>
- **Licence:** CC BY-NC-ND 4.0 (<https://github.com/ThunderVVV/HaWoR/blob/main/license.txt>)
- **Fetch:** scripts/fetch_weights.py --yes --only hawor --hawor-root <HaWoR checkout>; it also verifies mano_mean_params.npz, which the HaWoR repository tracks

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `hawor.ckpt` | 3,267,481,572 | `4d1cc43853c190d6f2c10d9b6295c73109f0faf9ef41ac817a2b31d94b4823f2` | `$FASTDO_HAWOR_ROOT/weights/hawor/checkpoints/hawor.ckpt` |
| `model_config.yaml` | 2,743 | `edfe12dc14ce371d698da722b59acfed5b4a38a7f8f5116cbc1fce459a07dd2d` | `$FASTDO_HAWOR_ROOT/weights/hawor/model_config.yaml` |
| `mano_mean_params.npz` | 1,178 | `efc0ec58e4a5cef78f3abfb4e8f91623b8950be9eff8b8e0dbb0d036ebc63988` | `$FASTDO_HAWOR_ROOT/_DATA/data/mano_mean_params.npz` |

### `wilor_detector`: WiLoR hand detector (YOLO), loaded by the HaWoR driver

- **Version:** HF space rolpotamias/WiLoR revision 99fe3d7acff8104ecca1055df7467709506c2fa6
- **Code:** ultralytics 8.4.36 (hawor env)
- **Source:** <https://huggingface.co/spaces/rolpotamias/WiLoR>
- **Licence:** CC BY-NC-ND 4.0 (stricter of two conflicting upstream statements; the other is CC BY-NC 2.0) (<https://github.com/rolpotamias/WiLoR/blob/main/license.txt>)
- **Fetch:** scripts/fetch_weights.py --yes --only wilor_detector --hawor-root <HaWoR checkout>

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `detector.pt` | 53,582,271 | `5ef3df44e42d2db52d4ffe91f83a22ce9925e2acc9abebf453f2c5d22e380033` | `$FASTDO_HAWOR_ROOT/weights/external/detector.pt` |

### `hands23`: Hands23 hand-object detector (opt-in internet-video prompt stage)

- **Version:** unversioned URL; code hands23_detector @ 1bc0f919ffa7a9f375e7e8042e1d26f3743d5819, detectron2 @ a2f4a8771ab77e8411c26b27f24f9489a28a2453
- **Code:** https://github.com/EvaCheng-cty/hands23_detector @ 1bc0f919ffa7a9f375e7e8042e1d26f3743d5819
- **Source:** <https://fouheylab.eecs.umich.edu/~dandans/projects/hands23/model_weights/model_hands23.pth>
- **Licence:** code MIT; weights carry no licence text; training data CC BY-NC 4.0 (<https://github.com/EvaCheng-cty/hands23_detector>)
- **Fetch:** scripts/fetch_weights.py --yes --only hands23 --hands23-root <dir>

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `model_hands23.pth` | 467,798,701 | `b3d2ff966d8a19b991a3ce170e2d8f86040f614ef2edf4a55423288b278f723c` | `$FASTDO_HANDS23_ROOT/weights/model_hands23.pth` |

### `mano`: MANO hand model (right and left)

- **Version:** the MANO_RIGHT.pkl / MANO_LEFT.pkl of the MANO download; identify yours by sha256
- **Code:** smplx 0.1.28 (hawor env)
- **Source:** <https://mano.is.tue.mpg.de>
- **Licence:** MANO licence (non-commercial; no redistribution in any form; no training of commercial models) (<https://mano.is.tue.mpg.de/license.html>) **GATED**
- **Fetch:** MANUAL ONLY: register, accept the licence, download it yourself; never fetched or shipped

| file | bytes | sha256 | expected at |
|---|---|---|---|
| `MANO_RIGHT.pkl` | 3,821,356 | `45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767` | `$FASTDO_MANO_DIR/MANO_RIGHT.pkl and $FASTDO_HAWOR_ROOT/_DATA/data/mano/MANO_RIGHT.pkl` |
| `MANO_LEFT.pkl` | 3,821,391 | `c4022f7083f2ca7c78b2b3d595abbab52debd32b09d372b16923a801f0ea6a30` | `$FASTDO_MANO_LEFT_DIR/MANO_LEFT.pkl and $FASTDO_HAWOR_ROOT/_DATA/data_left/mano_left/MANO_LEFT.pkl` |
