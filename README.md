# Native and Compact Structured Latents for 3D Generation

<a href="https://arxiv.org/abs/2512.14692"><img src="https://img.shields.io/badge/Paper-Arxiv-b31b1b.svg" alt="Paper"></a>
<a href="https://huggingface.co/microsoft/TRELLIS.2-4B"><img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow" alt="Hugging Face"></a>
<a href="https://huggingface.co/spaces/microsoft/TRELLIS.2"><img src="https://img.shields.io/badge/Hugging%20Face-Demo-blueviolet" alt="Hugging Face Demo"></a>
<a href="https://microsoft.github.io/TRELLIS.2"><img src="https://img.shields.io/badge/Project-Website-blue" alt="Project Page"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>

**TRELLIS.2** is a state-of-the-art large 3D generative model (4B parameters) designed for high-fidelity **image-to-3D** generation. It leverages a novel "field-free" sparse voxel structure termed **O-Voxel** to reconstruct and generate arbitrary 3D assets with complex topologies, sharp features, and full PBR materials.

This repository contains the **complete training codebase**: the sparse and compact SC-VAEs, the three flow models (sparse structure, shape, texture), the dataset pipeline, and a ready-to-run **Modal.com integration that trains the whole stack on a single NVIDIA B300 (Blackwell Ultra)**.

---

## 📁 Repository layout

| Path | Content |
| :--- | :--- |
| `train.py` | Training entry point (single GPU or distributed). |
| `modal_train.py` | Modal.com app: builds the CUDA 13.1 image, verifies the stack and runs resumable training on one B300. |
| `configs/scvae/` | SC-VAE configs (shape and PBR texture autoencoders, 256³ and 512³). |
| `configs/gen/` | Flow model configs (sparse structure, shape flow, texture flow). |
| `trellis2/` | The library: datasets, models, modules (sparse ops, attention), trainers, renderers. |
| `data_toolkit/` | Dataset preparation pipeline (download, mesh/PBR dumps, O-Voxel conversion, latent encoding). |
| `o-voxel/` | O-Voxel representation library (CUDA, compiled at install time). |
| `setup.sh` | Local (bare metal) dependency installer. |

---

## 🏋️ Training on Modal.com with a single B300

The B300 is a Blackwell Ultra GPU (**compute capability 10.3 / `sm_103`**) and requires a CUDA 13.1+ toolchain. `modal_train.py` takes care of every part of that stack:

* base image `nvidia/cuda:13.1.2-devel-ubuntu24.04` (Modal's host driver exposes CUDA 13.3),
* `torch` + `torchvision` from the **cu130** index (the only wheels that ship `sm_103` kernels),
* `flex_gemm`, `CuMesh`, `nvdiffrast`, `nvdiffrec` and `o_voxel` compiled from source for `TORCH_CUDA_ARCH_LIST="10.0;10.3"`,
* **`sdpa` attention backend** (`ATTN_BACKEND=sdpa`) so neither `flash-attn` nor `xformers` has to be built for a brand new architecture: PyTorch's fused `scaled_dot_product_attention` (cuDNN/flash kernels) is used for the dense transformer blocks *and* for the variable-length sparse attention of the flow models,
* one Modal volume with `datasets/`, `results/` and `hf-cache/`,
* checkpoint discovery, integrity checking and automatic resume after a preemption, a timeout or a crash.

### 1. One-time setup

```sh
pip install modal
modal setup                                          # authenticate the CLI
modal secret create huggingface HF_TOKEN=hf_...      # needed for the gated DINOv3 conditioner
modal volume create trellis2                         # or let `create_if_missing` do it
```

The first two commands prepare the local client, the secret is optional for SC-VAE training but required by every image-conditioned flow config (`facebook/dinov3-vitl16-pretrain-lvd1689m` is a gated repository).

### 2. Verify the software stack on a real B300 (strongly recommended)

```sh
modal run modal_train.py::verify
```

`verify` runs a forward (and mostly backward) pass through every kernel family the training loop uses — flex_gemm sparse convolutions, dense/sparse/windowed attention against a plain PyTorch reference, o_voxel hash + `.vxz` round trip, nvdiffrast rasterisation, nvdiffrec environment lighting, CuMesh, the `MeshRenderer` used by the SC-VAE losses and the tensorboard writer — and prints a JSON report (GPU name, capability, HBM, torch/CUDA versions, `torch.cuda.get_arch_list()`). It takes a couple of minutes and fails loudly instead of letting a long run die on a missing `sm_103` kernel.

### 3. Put the dataset in the volume

The dataset must be prepared first, see [data_toolkit/README.md](data_toolkit/README.md). Then copy the prepared root into the volume:

```sh
modal volume put trellis2 ./datasets/MyDataset /datasets/MyDataset
```

Inside the container the roots are available under `/vol/datasets` (e.g. `/vol/datasets/MyDataset`), the training outputs are written to `/vol/results/<run_name>`.

### 4. Warm the Hugging Face cache (optional, saves a few minutes)

```sh
modal run modal_train.py::prefetch_weights
```

Downloads `microsoft/TRELLIS.2-4B`, `JeffreyXiang/TRELLIS.2-4B`, `JeffreyXiang/TRELLIS-image-large` and DINOv3 into `/vol/hf-cache`.

### 5. Launch a training run

Shape SC-VAE at 256³:

```sh
modal run --detach modal_train.py \
  --config configs/scvae/shape_vae_next_dc_f16c32_fp16.json \
  --data-dir '{"MyDataset": {"base": "/vol/datasets/MyDataset", "mesh_dump": "/vol/datasets/MyDataset/mesh_dumps", "dual_grid": "/vol/datasets/MyDataset/dual_grid_256", "asset_stats": "/vol/datasets/MyDataset/asset_stats"}}' \
  --run-name shape_vae_256
```

Texture SC-VAE at 256³:

```sh
modal run --detach modal_train.py \
  --config configs/scvae/tex_vae_next_dc_f16c32_fp16.json \
  --data-dir '{"MyDataset": {"base": "/vol/datasets/MyDataset", "pbr_dump": "/vol/datasets/MyDataset/pbr_dumps", "pbr_voxel": "/vol/datasets/MyDataset/pbr_voxels_256", "asset_stats": "/vol/datasets/MyDataset/asset_stats"}}' \
  --run-name tex_vae_256
```

Sparse structure flow model:

```sh
modal run --detach modal_train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_64_bf16.json \
  --data-dir '{"MyDataset": {"base": "/vol/datasets/MyDataset", "ss_latent": "/vol/datasets/MyDataset/ss_latents/ss_enc_conv3d_16l8_fp16_64", "render_cond": "/vol/datasets/MyDataset/renders_cond"}}' \
  --run-name ss_flow
```

Shape flow model:

```sh
modal run --detach modal_train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json \
  --data-dir '{"MyDataset": {"base": "/vol/datasets/MyDataset", "shape_latent": "/vol/datasets/MyDataset/shape_latents/shape_enc_next_dc_f16c32_fp16_512", "render_cond": "/vol/datasets/MyDataset/renders_cond"}}' \
  --run-name shape_flow
```

Texture flow model:

```sh
modal run --detach modal_train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json \
  --data-dir '{"MyDataset": {"base": "/vol/datasets/MyDataset", "shape_latent": "/vol/datasets/MyDataset/shape_latents/shape_enc_next_dc_f16c32_fp16_512", "pbr_latent": "/vol/datasets/MyDataset/pbr_latents/tex_enc_next_dc_f16c32_fp16_512", "render_cond": "/vol/datasets/MyDataset/renders_cond"}}' \
  --run-name tex_flow
```

`--detach` submits the job and returns immediately; the training keeps running on Modal. Without it the CLI blocks and streams the return value. Follow the logs with:

```sh
modal app logs trellis2-train
```

### 6. Command line reference of `modal_train.py`

| Flag | Meaning |
| :--- | :--- |
| `--config` | Config path relative to the repository root (default: the 256³ shape SC-VAE). |
| `--data-dir` | Dataset directory or the JSON dataset specification of `train.py` (required). |
| `--run-name` | Output directory inside `/vol/results` (default: `<config stem>-<timestamp>`). |
| `--overrides` | JSON object deep-merged on top of the config file, e.g. `'{"dataset": {"args": {"resolution": 512}}}'`. |
| `--ckpt` | `latest` (default, resumes from the newest *complete* checkpoint), `none` (fresh start) or an explicit step. |
| `--max-steps` | Override `trainer.args.max_steps` when > 0. |
| `--batch-size` | Override `trainer.args.batch_size_per_gpu` when > 0. |
| `--i-save` | Override `trainer.args.i_save` (default 200 steps, 0 keeps the config value). |
| `--detach/--no-detach` | Submit and return instead of blocking. |
| `--verify-first` | Run the stack verification before submitting the job. |
| `--extra-train-args` | Extra whitespace separated flags appended to the `train.py` command line. |

Every run writes the fully resolved config to `<output_dir>/config_resolved.json` next to the upstream `command.txt` / `config.json`, so an experiment is always reproducible.

### 7. What happens on a preemption, timeout or crash

* Checkpoints (`ckpts/*_step*.pt` plus `misc_step*.pt`) are flushed to the volume every 5 minutes by a background committer and immediately on `SIGTERM` (Modal sends one when it preempts a container).
* The next attempt scans `ckpts/misc_step*.pt`, validates the newest candidates with `zipfile` (an interrupted `torch.save` leaves a broken archive) and resumes from the newest step whose model, EMA and optimizer files are all readable. Nothing is silently lost and no corrupt checkpoint is ever loaded.
* Modal restarts preempted functions automatically; failures are retried up to `TRELLIS_MAX_RETRIES` (default 20) with exponential backoff, always resuming from the newest checkpoint.
* A single function call is capped at 24 h (`TRELLIS_MAX_RUNTIME_SECONDS`); a run that needs longer simply continues on the next call.

### 8. Tuning for one B300

A B300 has 262 GB of HBM (roughly 3× an H100-80GB), so the per-GPU batch size of the shipped configs — which were authored for 8×80 GB — can be raised:

```sh
# 256³ SC-VAE: 8 (config) -> 24 works comfortably on 262 GB
modal run --detach modal_train.py --config configs/scvae/shape_vae_next_dc_f16c32_fp16.json ... --batch-size 24

# 512³ SC-VAE and the flow models: keep the config values, they are memory bound by the
# 512³/1024³ decoders and by the elastic memory controllers.
```

Useful environment variables (read on the deploying machine, so `export` them before `modal run`):

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `TRELLIS_GPU` | `B300` | GPU request (`B300` = one B300; `B300:4` would request four). |
| `TRELLIS_VOLUME_NAME` | `trellis2` | Modal volume holding data, results and the HF cache. |
| `TRELLIS_DATALOADER_WORKERS` | `24` | Dataloader workers (see `DATALOADER_NUM_WORKERS` below). |
| `TRELLIS_CPU` / `TRELLIS_MEMORY_MB` | `32` / `131072` | CPU cores and host RAM of the training container. |
| `TRELLIS_I_SAVE` | `200` | Default checkpoint interval. |
| `TRELLIS_MAX_RETRIES` | `20` | Retries before giving up on a crashing run. |
| `TRELLIS_MAX_RUNTIME_SECONDS` | `86400` | Hard cap of a single container (24 h). |
| `TRELLIS_COMMIT_INTERVAL_SECONDS` | `300` | Volume commit interval. |
| `TRELLIS_TORCH_SPEC` | `torch torchvision` | Pin the torch pair, e.g. `"torch==2.14.1 torchvision==0.29.1"`. |
| `TRELLIS_ATTN_BACKEND` | `sdpa` | `sdpa`, `xformers`, `flash_attn` or `flash_attn_3`. |
| `TRELLIS_CONV_BACKEND` | `flex_gemm` | `flex_gemm`, `spconv`, `torchsparse` or `none`. |
| `TRELLIS_ENVMAP` | unset | Path to an equirectangular `.exr`/`.hdr` used by the PBR sample visualisations (a procedural studio environment is used otherwise). |

### 9. Troubleshooting

* **`no kernel image is available for execution on the device`** — an extension was compiled for the wrong architecture. Rebuild the image with `TRELLIS_TORCH_CUDA_ARCH_LIST="10.3"` (or `"10.0;10.3"`).
* **`ImportError: libcudart.so.13` / driver mismatch** — the base image must stay ≤ the host CUDA driver (Modal exposes CUDA 13.3, so `13.1.x` and `13.2.x` images work).
* **CUDA out of memory** — lower `--batch-size`, or lower `trainer.args.batch_split` (fewer micro batches means a larger activation peak, so raise it if the config allows).
* **The volume copy is slow** — upload once with `modal volume put`, and keep the number of small files per directory moderate; the dataloaders stream from the volume over the network.
* **Gated model errors from `transformers`** — the `huggingface` secret is missing or its token does not have access to DINOv3; run `modal secret create huggingface HF_TOKEN=hf_...` and redeploy.

---

## 🛠️ Local installation (without Modal)

### Prerequisites

- **System**: The code is currently tested only on **Linux**.
- **Hardware**: An NVIDIA GPU with at least 24GB of memory is necessary. The code has been verified on NVIDIA A100 and H100 GPUs.
- **Software**:
  - The [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) is needed to compile certain packages. Recommended version is 12.4. For Blackwell (B200/B300) use CUDA 13.1+ and the cu130 torch wheels instead.
  - [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) is recommended for managing dependencies.
  - Python version 3.8 or higher is required.

### Installation steps

1. Clone the repo:
    ```sh
    git clone -b main https://github.com/microsoft/TRELLIS.2.git --recursive
    cd TRELLIS.2
    ```

2. Install the dependencies:

    **Before running the following command there are somethings to note:**
    - By adding `--new-env`, a new conda environment named `trellis2` will be created. If you want to use an existing conda environment, please remove this flag.
    - By default, the `trellis2` environment will use pytorch 2.6.0 with CUDA 12.4. If you want to use a different version of CUDA, you can remove the `--new-env` flag and manually install the required dependencies. Refer to [PyTorch](https://pytorch.org/get-started/previous-versions/) for the installation command.
    - If you have multiple CUDA Toolkit versions installed, `CUDA_HOME` should be set to the correct version before running the command. For example, if you have CUDA Toolkit 12.4 and 13.0 installed, you can run `export CUDA_HOME=/usr/local/cuda-12.4` before running the command.
    - By default, the code uses the `flash-attn` backend for attention. For GPUs that do not have prebuilt `flash-attn` binaries (e.g. NVIDIA B300), set `ATTN_BACKEND=sdpa` — the exact variable-length implementation shipped in `trellis2/modules/sparse/attention` — or install `xformers` and set `ATTN_BACKEND=xformers`.
    - The installation may take a while due to the large number of dependencies. Please be patient. If you encounter any issues, you can try to install the dependencies one by one, specifying one flag at a time.
    - The `o-voxel/third_party/eigen` submodule lives on gitlab.com; `modal_train.py` replaces it with the packaged Eigen headers, and the same trick works locally (`sudo apt install libeigen3-dev && ln -s /usr/include/eigen3 o-voxel/third_party/eigen`).

    Create a new conda environment named `trellis2` and install the dependencies:
    ```sh
    . ./setup.sh --new-env --basic --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel --flexgemm
    ```
    The detailed usage of `setup.sh` can be found by running `. ./setup.sh --help`.

---

## 🏋️ Local training

### 1. Data preparation

Before training, raw 3D assets must be converted into the **O-Voxel** representation. This process includes mesh conversion, compact structured latent generation, and metadata preparation.

> 📂 **Please refer to [data_toolkit/README.md](data_toolkit/README.md) for detailed instructions on data preprocessing and dataset organization.**

### 2. Running training

Training is managed through the `train.py` script, which accepts multiple command-line arguments to configure experiments:

* `--config`: Path to the experiment configuration file.
* `--output_dir`: Directory for training outputs.
* `--load_dir`: Directory to load checkpoints from (defaults to `output_dir`).
* `--ckpt`: Checkpoint step to resume from (defaults to the latest).
* `--data_dir`: Dataset path or a JSON string specifying dataset locations.
* `--auto_retry`: Number of automatic retries upon failure.
* `--tryrun`: Perform a dry run without actual training.
* `--profile`: Enable training profiling.
* `--num_nodes`: Number of nodes for distributed training.
* `--node_rank`: Rank of the current node.
* `--num_gpus`: Number of GPUs per node (defaults to all available GPUs).
* `--master_addr`: Master node address for distributed training.
* `--master_port`: Port for distributed training communication.

The dataloader worker count can be pinned with the `DATALOADER_NUM_WORKERS` environment variable; without it the trainer uses the number of CPUs available to the process divided by the number of visible GPUs.

### SC-VAE training

To train the shape SC-VAE, run:

```sh
python train.py \
  --config configs/scvae/shape_vae_next_dc_f16c32_fp16.json \
  --output_dir results/shape_vae_next_dc_f16c32_fp16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"mesh_dump\": \"datasets/ObjaverseXL_sketchfab/mesh_dumps\", \"dual_grid\": \"datasets/ObjaverseXL_sketchfab/dual_grid_256\", \"asset_stats\": \"datasets/ObjaverseXL_sketchfab/asset_stats\"}}"
```

This command trains the shape SC-VAE on the **Objaverse-XL** dataset using the `shape_vae_next_dc_f16c32_fp16.json` configuration. Training outputs will be saved to `results/shape_vae_next_dc_f16c32_fp16`.

The dataset is specified as a JSON string, where each dataset entry includes:

* `base`: Root directory of the dataset.
* `mesh_dump`: Directory containing preprocessed mesh dumps.
* `dual_grid`: Directory with precomputed dual-grid representations.
* `asset_stats`: Directory containing precomputed asset statistics.

To fine-tune the model at a higher resolution, use the `shape_vae_next_dc_f16c32_fp16_ft_512.json` configuration. Remember to update the `finetune_ckpt` field and adjust the dataset paths accordingly.

To train the texture SC-VAE, run:

```sh
python train.py \
  --config configs/scvae/tex_vae_next_dc_f16c32_fp16.json \
  --output_dir results/tex_vae_next_dc_f16c32_fp16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"pbr_dump\": \"datasets/ObjaverseXL_sketchfab/pbr_dumps\", \"pbr_voxel\": \"datasets/ObjaverseXL_sketchfab/pbr_voxels_256\", \"asset_stats\": \"datasets/ObjaverseXL_sketchfab/asset_stats\"}}"
```

### Flow model training

To train the sparse structure flow model, run:

```sh
python train.py \
  --config configs/gen/ss_flow_img_dit_1_3B_64_bf16.json \
  --output_dir results/ss_flow_img_dit_1_3B_64_bf16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"ss_latent\": \"datasets/ObjaverseXL_sketchfab/ss_latents/ss_enc_conv3d_16l8_fp16_64\", \"render_cond\": \"datasets/ObjaverseXL_sketchfab/renders_cond\"}}"
```

This command trains the sparse-structure flow model on the **Objaverse-XL** dataset using the specified configuration file. Outputs are saved to `results/ss_flow_img_dit_1_3B_64_bf16`.

The dataset configuration includes:

* `base`: Root dataset directory.
* `ss_latent`: Directory containing precomputed sparse-structure latents.
* `render_cond`: Directory containing conditional rendering images.

The second- and third-stage flow models for shape and texture generation can be trained using the following configurations:

* Shape flow: `slat_flow_img2shape_dit_1_3B_512_bf16.json`
* Texture flow: `slat_flow_imgshape2tex_dit_1_3B_512_bf16.json`

Example commands:

```sh
# Shape flow model
python train.py \
  --config configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json \
  --output_dir results/slat_flow_img2shape_dit_1_3B_512_bf16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"shape_latent\": \"datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512\", \"render_cond\": \"datasets/ObjaverseXL_sketchfab/renders_cond\"}}"

# Texture flow model
python train.py \
  --config configs/gen/slat_flow_imgshape2tex_dit_1_3B_512_bf16.json \
  --output_dir results/slat_flow_imgshape2tex_dit_1_3B_512_bf16 \
  --data_dir "{\"ObjaverseXL_sketchfab\": {\"base\": \"datasets/ObjaverseXL_sketchfab\", \"shape_latent\": \"datasets/ObjaverseXL_sketchfab/shape_latents/shape_enc_next_dc_f16c32_fp16_512\", \"pbr_latent\": \"datasets/ObjaverseXL_sketchfab/pbr_latents/tex_enc_next_dc_f16c32_fp16_512\", \"render_cond\": \"datasets/ObjaverseXL_sketchfab/renders_cond\"}}"
```

Higher-resolution fine-tuning can be performed by updating the `finetune_ckpt` field in the following configuration files and adjusting the dataset paths accordingly:

* `slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json`
* `slat_flow_imgshape2tex_dit_1_3B_512_bf16_ft1024.json`

### Attention and convolution backends

`trellis2/modules/attention` (dense) and `trellis2/modules/sparse/attention` (variable-length) support four interchangeable backends, selected with `ATTN_BACKEND` / `SPARSE_ATTN_BACKEND`:

* `sdpa` (default in `modal_train.py`) — PyTorch `scaled_dot_product_attention`; exact variable-length attention without padding, no third party build, works on every new architecture.
* `flash_attn` / `flash_attn_3` — fastest on Hopper, requires the corresponding prebuilt wheel.
* `xformers` — `memory_efficient_attention` with block diagonal masks.

Sparse convolutions are dispatched through `SPARSE_CONV_BACKEND` (`flex_gemm` by default, plus `spconv`, `torchsparse` and `none`).

---

## 🧩 Related packages

TRELLIS.2 is built upon several specialized high-performance packages developed by our team:

*   **[O-Voxel](o-voxel):** Core library handling the logic for converting between textured meshes and the O-Voxel representation, ensuring instant bidirectional transformation.
*   **[FlexGEMM](https://github.com/JeffreyXiang/FlexGEMM):** Efficient sparse convolution implementation based on Triton, enabling rapid processing of sparse voxel structures.
*   **[CuMesh](https://github.com/JeffreyXiang/CuMesh):** CUDA-accelerated mesh utilities used for high-speed post-processing, remeshing, decimation, and UV-unwrapping.

---

## ⚖️ License

This model and code are released under the **[MIT License](LICENSE)**.

Please note that certain dependencies operate under separate license terms:

- [**nvdiffrast**](https://github.com/NVlabs/nvdiffrast): Utilized as the rasterisation backend of the SC-VAE rendering losses. This package is governed by its own [License](https://github.com/NVlabs/nvdiffrast/blob/main/LICENSE.txt).

- [**nvdiffrec**](https://github.com/NVlabs/nvdiffrec): Implements the split-sum renderer for PBR materials. This package is governed by its own [License](https://github.com/NVlabs/nvdiffrec/blob/main/LICENSE.txt).

## 📚 Citation

If you find this model useful for your research, please cite our work:

```bibtex
@article{
    xiang2025trellis2,
    title={Native and Compact Structured Latents for 3D Generation},
    author={Xiang, Jianfeng and Chen, Xiaoxue and Xu, Sicheng and Wang, Ruicheng and Lv, Zelong and Deng, Yu and Zhu, Hongyuan and Dong, Yue and Zhao, Hao and Yuan, Nicholas Jing and Yang, Jiaolong},
    journal={Tech report},
    year={2025}
}
```
