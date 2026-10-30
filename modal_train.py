from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import modal

APP_NAME = os.environ.get("TRELLIS_APP_NAME", "trellis2-train")

GPU_SPEC = os.environ.get("TRELLIS_GPU", "B300")
GPU_COUNT = 1

CUDA_IMAGE = os.environ.get("TRELLIS_CUDA_IMAGE", "nvidia/cuda:13.1.2-devel-ubuntu24.04")
PYTHON_VERSION = os.environ.get("TRELLIS_PYTHON_VERSION", "3.12")

TORCH_INDEX_URL = os.environ.get("TRELLIS_TORCH_INDEX_URL", "https://download.pytorch.org/whl/cu130")
TORCH_SPEC = os.environ.get("TRELLIS_TORCH_SPEC", "torch torchvision")

TORCH_CUDA_ARCH_LIST = os.environ.get("TRELLIS_TORCH_CUDA_ARCH_LIST", "10.0;10.3")

TRANSFORMERS_VERSION = os.environ.get("TRELLIS_TRANSFORMERS_VERSION", "4.57.6")

UTILS3D_COMMIT = os.environ.get("TRELLIS_UTILS3D_COMMIT", "9a4eb15e4021b67b12c460c7057d642626897ec8")

VOLUME_NAME = os.environ.get("TRELLIS_VOLUME_NAME", "trellis2")
VOLUME_MOUNT = "/vol"
DATA_ROOT = f"{VOLUME_MOUNT}/datasets"
RESULTS_ROOT = f"{VOLUME_MOUNT}/results"
HF_CACHE = f"{VOLUME_MOUNT}/hf-cache"

REPO_REMOTE_PATH = "/opt/trellis2"
BUILD_DIR = "/opt/build"
LOCAL_REPO_ROOT = Path(__file__).resolve().parent

ATTENTION_BACKEND = os.environ.get("TRELLIS_ATTN_BACKEND", "sdpa")
SPARSE_CONV_BACKEND = os.environ.get("TRELLIS_CONV_BACKEND", "flex_gemm")

DATALOADER_WORKERS = int(os.environ.get("TRELLIS_DATALOADER_WORKERS", "24"))

HF_SECRET_NAME = os.environ.get("TRELLIS_HF_SECRET_NAME", "huggingface")

MAX_RUNTIME_SECONDS = int(os.environ.get("TRELLIS_MAX_RUNTIME_SECONDS", str(24 * 60 * 60)))
MAX_RETRIES = int(os.environ.get("TRELLIS_MAX_RETRIES", "20"))
COMMIT_INTERVAL_SECONDS = int(os.environ.get("TRELLIS_COMMIT_INTERVAL_SECONDS", "300"))

CPU_REQUEST = float(os.environ.get("TRELLIS_CPU", "32"))
MEMORY_REQUEST_MB = int(os.environ.get("TRELLIS_MEMORY_MB", str(128 * 1024)))

DEFAULT_I_SAVE = int(os.environ.get("TRELLIS_I_SAVE", "200"))

PRETRAINED_REPOSITORIES = (
    ("microsoft/TRELLIS.2-4B", ["ckpts/*", "*.json", "*.md"]),
    ("JeffreyXiang/TRELLIS.2-4B", ["ckpts/*", "*.json", "*.md"]),
    ("JeffreyXiang/TRELLIS-image-large", ["ckpts/*", "*.json", "*.md"]),
    ("facebook/dinov3-vitl16-pretrain-lvd1689m", ["*"]),
)

PYTHON_PACKAGES = (
    "easydict",
    "filelock",
    "huggingface_hub[hf_xet]",
    "imageio",
    "imageio-ffmpeg",
    "lpips",
    "matplotlib",
    "ninja",
    "numpy",
    "opencv-python-headless",
    "pandas",
    "pillow",
    "plyfile",
    "requests",
    "safetensors",
    "tensorboard",
    "tqdm",
    "trimesh",
    "zstandard",
    f"transformers=={TRANSFORMERS_VERSION}",
)

APT_PACKAGES = (
    "build-essential",
    "ca-certificates",
    "cmake",
    "curl",
    "git",
    "libeigen3-dev",
    "libgl1",
    "libglib2.0-0",
    "ninja-build",
)

BASE_ENV = {
    "CUDA_HOME": "/usr/local/cuda",
    "FORCE_CUDA": "1",
    "MAX_JOBS": os.environ.get("TRELLIS_MAX_JOBS", "16"),
    "TORCH_CUDA_ARCH_LIST": TORCH_CUDA_ARCH_LIST,
    "ATTN_BACKEND": ATTENTION_BACKEND,
    "SPARSE_ATTN_BACKEND": ATTENTION_BACKEND,
    "SPARSE_CONV_BACKEND": SPARSE_CONV_BACKEND,
    "OPENCV_IO_ENABLE_OPENEXR": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYTHONUNBUFFERED": "1",
    "OMP_NUM_THREADS": os.environ.get("TRELLIS_OMP_NUM_THREADS", "8"),
}

RUNTIME_ENV = {
    "HF_HOME": HF_CACHE,
    "HF_HUB_CACHE": f"{HF_CACHE}/hub",
    "TORCH_HOME": f"{HF_CACHE}/torch",
    "DATALOADER_NUM_WORKERS": str(DATALOADER_WORKERS),
    "PYTHONPATH": REPO_REMOTE_PATH,
}

MISC_CHECKPOINT_PATTERN = re.compile(r"^misc_step(\d+)\.pt$")

LOCAL_PATHS_SENT_TO_THE_CONTAINER_TO_IGNORE = (
    ".git/**",
    ".github/**",
    ".venv/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
    "results/**",
    "datasets/**",
    "data/**",
    "logs/**",
    "**/*.pt",
    "**/*.pth",
    "**/*.safetensors",
    "**/*.mp4",
    "**/*.glb",
    "**/*.vxz",
    "**/*.npz",
    "**/*.exr",
)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = json.loads(json.dumps(base))
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _ema_rate_list(value: Any) -> List[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        return [float(rate) for rate in value]
    raise TypeError(f"Unsupported ema_rate value: {value!r}")


def _checkpoint_file_is_readable(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return len(archive.namelist()) > 0
    except (zipfile.BadZipFile, EOFError, OSError, ValueError):
        return False


def _find_resumable_step(
    ckpts_dir: Path,
    model_names: Sequence[str],
    ema_rates: Sequence[float],
) -> Optional[int]:
    if not ckpts_dir.is_dir():
        return None

    steps: List[int] = []
    for entry in ckpts_dir.iterdir():
        match = MISC_CHECKPOINT_PATTERN.match(entry.name)
        if match is not None:
            steps.append(int(match.group(1)))

    for step in sorted(steps, reverse=True):
        required = [ckpts_dir / f"misc_step{step:07d}.pt"]
        for name in model_names:
            required.append(ckpts_dir / f"{name}_step{step:07d}.pt")
        for ema_rate in ema_rates:
            for name in model_names:
                required.append(ckpts_dir / f"{name}_ema{ema_rate}_step{step:07d}.pt")

        unreadable = [path.name for path in required if not _checkpoint_file_is_readable(path)]
        if unreadable:
            print(
                f"[modal] step {step} is unusable, skipping it because of: {', '.join(unreadable)}",
                flush=True,
            )
            continue
        return step

    return None


def _dataset_paths(data_dir: str) -> List[str]:
    try:
        parsed = json.loads(data_dir)
    except json.JSONDecodeError:
        parsed = data_dir

    collected: List[str] = []

    def _walk(node: Any) -> None:
        if isinstance(node, str):
            collected.extend(part for part in node.split(",") if part)
        elif isinstance(node, dict):
            for value in node.values():
                _walk(value)

    _walk(parsed)
    return collected


def _validate_data_dir(data_dir: str) -> None:
    paths = _dataset_paths(data_dir)
    if not paths:
        raise ValueError(
            "No dataset path found in --data-dir. Pass either a directory or the JSON object "
            'documented in data_toolkit/README.md, e.g. \'{"MyDataset": {"base": "/vol/datasets/MyDataset"}}\''
        )
    missing = [path for path in paths if not Path(path).exists()]
    if missing:
        readable = "\n  ".join(missing)
        raise FileNotFoundError(
            "The following dataset paths are not visible inside the container:\n  "
            f"{readable}\n"
            f"Upload the dataset with `modal volume put {VOLUME_NAME} ./datasets/MyDataset "
            f"{DATA_ROOT}/MyDataset` and reference it with absolute {DATA_ROOT} paths."
        )


def _resolve_config_path(config: str) -> Path:
    candidate = Path(config)
    if candidate.is_absolute():
        resolved = candidate
    else:
        resolved = Path(REPO_REMOTE_PATH) / candidate
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Config file {config!r} not found (looked for {resolved}). "
            "Pass a path relative to the repository root, e.g. configs/scvae/"
            "shape_vae_next_dc_f16c32_fp16.json"
        )
    return resolved


def _hf_secrets() -> List[modal.Secret]:
    try:
        available_secrets = {secret.name for secret in modal.Secret.objects.list()}
    except Exception as error:
        print(
            f"[modal] could not list the workspace secrets ({error}), continuing without "
            f"the {HF_SECRET_NAME!r} secret",
            flush=True,
        )
        return []
    if HF_SECRET_NAME not in available_secrets:
        print(
            f"[modal] warning: the {HF_SECRET_NAME!r} secret is not defined, so gated "
            "checkpoints (DINOv3) cannot be downloaded. Create it with: "
            "modal secret create huggingface HF_TOKEN=hf_...",
            flush=True,
        )
        return []
    return [modal.Secret.from_name(HF_SECRET_NAME, required_keys=["HF_TOKEN"])]


def _nvidia_smi_summary() -> str:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return proc.stdout.strip()


def _third_party_extension_commands() -> List[str]:
    return [
        f"mkdir -p {BUILD_DIR}",
        f"git clone --depth 1 --branch v0.4.0 https://github.com/NVlabs/nvdiffrast.git {BUILD_DIR}/nvdiffrast",
        f"git clone --depth 1 --branch renderutils https://github.com/JeffreyXiang/nvdiffrec.git {BUILD_DIR}/nvdiffrec",
        f"git clone --depth 1 --recursive https://github.com/JeffreyXiang/CuMesh.git {BUILD_DIR}/CuMesh",
        f"git clone --depth 1 --recursive https://github.com/JeffreyXiang/FlexGEMM.git {BUILD_DIR}/FlexGEMM",
        f"python -m pip install --no-cache-dir --no-build-isolation {BUILD_DIR}/nvdiffrast",
        f"python -m pip install --no-cache-dir --no-build-isolation {BUILD_DIR}/nvdiffrec",
        f"python -m pip install --no-cache-dir --no-build-isolation {BUILD_DIR}/CuMesh",
        f"python -m pip install --no-cache-dir --no-build-isolation {BUILD_DIR}/FlexGEMM",
        "python -m pip install --no-cache-dir "
        f"'git+https://github.com/EasternJournalist/utils3d.git@{UTILS3D_COMMIT}'",
    ]


def _o_voxel_build_commands() -> List[str]:
    return [
        f"rm -rf {BUILD_DIR}/o-voxel/third_party/eigen",
        f"ln -s /usr/include/eigen3 {BUILD_DIR}/o-voxel/third_party/eigen",
        f"python -m pip install --no-cache-dir --no-build-isolation {BUILD_DIR}/o-voxel",
    ]


def _build_training_image() -> modal.Image:
    image = modal.Image.from_registry(CUDA_IMAGE, add_python=PYTHON_VERSION)
    image = image.entrypoint([])
    image = image.apt_install(*APT_PACKAGES)
    image = image.env(BASE_ENV)
    image = image.run_commands(
        "python -m pip install --no-cache-dir --upgrade pip setuptools wheel"
    )
    image = image.run_commands(
        f"python -m pip install --no-cache-dir {TORCH_SPEC} --index-url {TORCH_INDEX_URL}"
    )
    image = image.pip_install(*PYTHON_PACKAGES)
    image = image.run_commands(*_third_party_extension_commands())
    image = image.add_local_dir(
        str(LOCAL_REPO_ROOT / "o-voxel"),
        f"{BUILD_DIR}/o-voxel",
        copy=True,
    )
    image = image.run_commands(*_o_voxel_build_commands())
    image = image.add_local_dir(
        str(LOCAL_REPO_ROOT),
        REPO_REMOTE_PATH,
        copy=False,
        ignore=list(LOCAL_PATHS_SENT_TO_THE_CONTAINER_TO_IGNORE),
    )
    return image


training_image = _build_training_image()

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)


@app.function(
    image=training_image,
    gpu=GPU_SPEC,
    volumes={VOLUME_MOUNT: volume},
    timeout=MAX_RUNTIME_SECONDS,
    retries=modal.Retries(
        max_retries=MAX_RETRIES,
        initial_delay=60.0,
        backoff_coefficient=2.0,
        max_delay=60.0,
    ),
    cpu=CPU_REQUEST,
    memory=MEMORY_REQUEST_MB,
    max_containers=1,
    secrets=_hf_secrets(),
    env=RUNTIME_ENV,
)
def train(
    config: str,
    data_dir: str,
    output_dir: str,
    overrides: Optional[Dict[str, Any]] = None,
    ckpt: str = "latest",
    extra_args: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    os.chdir(REPO_REMOTE_PATH)
    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)

    started_at = time.time()
    volume.reload()

    gpu_summary = _nvidia_smi_summary()
    if gpu_summary:
        print(f"[modal] GPU: {gpu_summary}", flush=True)
    print(f"[modal] attention backend: {ATTENTION_BACKEND}, sparse conv backend: {SPARSE_CONV_BACKEND}", flush=True)

    config_path = _resolve_config_path(config)
    output_path = Path(output_dir)
    if not output_path.is_absolute():
        output_path = Path(RESULTS_ROOT) / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "ckpts").mkdir(parents=True, exist_ok=True)
    (output_path / "samples").mkdir(parents=True, exist_ok=True)

    raw_config = json.loads(config_path.read_text())
    resolved_config = _deep_merge(raw_config, overrides or {})
    resolved_config_path = output_path / "config_resolved.json"
    resolved_config_path.write_text(json.dumps(resolved_config, indent=4))

    _validate_data_dir(data_dir)
    print(f"[modal] config: {config_path} -> {resolved_config_path}", flush=True)
    print(f"[modal] output: {output_path}", flush=True)

    model_names = list(resolved_config.get("models", {}).keys())
    trainer_args = resolved_config.get("trainer", {}).get("args", {})
    ema_rates = _ema_rate_list(trainer_args.get("ema_rate", 0.9999))

    resumable_step = _find_resumable_step(output_path / "ckpts", model_names, ema_rates)
    if ckpt == "latest":
        ckpt_arg = "none" if resumable_step is None else str(resumable_step)
    else:
        ckpt_arg = ckpt
    print(f"[modal] resuming from checkpoint: {ckpt_arg}", flush=True)

    command = [
        sys.executable,
        "train.py",
        "--config",
        str(resolved_config_path),
        "--output_dir",
        str(output_path),
        "--load_dir",
        str(output_path),
        "--ckpt",
        ckpt_arg,
        "--data_dir",
        data_dir,
        "--num_gpus",
        str(GPU_COUNT),
        "--auto_retry",
        "0",
    ]
    command.extend(extra_args or [])

    stop_event = threading.Event()
    process_holder: Dict[str, Optional[subprocess.Popen]] = {"process": None}

    def _commit_volume(reason: str) -> None:
        try:
            volume.commit()
            print(f"[modal] volume committed ({reason})", flush=True)
        except Exception as error:
            print(f"[modal] volume commit failed ({reason}): {error}", flush=True)

    def _commit_loop() -> None:
        while not stop_event.wait(COMMIT_INTERVAL_SECONDS):
            _commit_volume("periodic")

    def _handle_signal(signum: int, _frame: Any) -> None:
        print(f"[modal] received signal {signum}, flushing checkpoints to the volume", flush=True)
        stop_event.set()
        process = process_holder["process"]
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                print("[modal] training process ignored SIGTERM, killing it", flush=True)
                process.kill()
                process.wait(timeout=60)
        _commit_volume("signal")

    previous_handlers = {}
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            previous_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, _handle_signal)
        except ValueError:
            print(
                f"[modal] cannot install a SIG{signum} handler outside the main thread, "
                "the periodic volume commit stays active",
                flush=True,
            )

    committer = threading.Thread(target=_commit_loop, name="volume-committer", daemon=True)
    committer.start()

    return_code = 1
    try:
        print(f"[modal] launching: {' '.join(command)}", flush=True)
        process = subprocess.Popen(
            command,
            cwd=REPO_REMOTE_PATH,
            env=os.environ.copy(),
            start_new_session=True,
        )
        process_holder["process"] = process
        return_code = process.wait()
        print(f"[modal] train.py exited with code {return_code}", flush=True)
    finally:
        stop_event.set()
        committer.join(timeout=60.0)
        _commit_volume("shutdown")
        for signum, handler in previous_handlers.items():
            try:
                signal.signal(signum, handler)
            except ValueError:
                pass

    if return_code != 0:
        raise RuntimeError(
            f"train.py exited with code {return_code}. The next attempt resumes from the newest "
            f"complete checkpoint in {output_path / 'ckpts'}."
        )

    return {
        "status": "finished",
        "config": str(config_path),
        "output_dir": str(output_path),
        "resumed_from_step": ckpt_arg,
        "duration_seconds": round(time.time() - started_at, 1),
        "gpu": gpu_summary,
    }


def _reference_attention(q: Any, k: Any, v: Any) -> Any:
    torch = sys.modules["torch"]
    scale = 1.0 / math.sqrt(q.shape[-1])
    weights = torch.softmax(torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale, dim=-1)
    return torch.matmul(weights, v.float())


@app.function(
    image=training_image,
    gpu=GPU_SPEC,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
    cpu=8.0,
    memory=32 * 1024,
    secrets=_hf_secrets(),
    env=RUNTIME_ENV,
)
def verify() -> Dict[str, Any]:
    os.chdir(REPO_REMOTE_PATH)
    if REPO_REMOTE_PATH not in sys.path:
        sys.path.insert(0, REPO_REMOTE_PATH)

    import torch

    import flex_gemm
    import o_voxel
    import cumesh
    import utils3d
    import nvdiffrast.torch as dr
    from nvdiffrec_render.light import EnvironmentLight
    from trellis2.modules import sparse as sp
    from trellis2.modules.attention.full_attn import scaled_dot_product_attention
    from trellis2.modules.sparse.attention.full_attn import sparse_scaled_dot_product_attention
    from trellis2.modules.sparse.attention.windowed_attn import (
        sparse_windowed_scaled_dot_product_self_attention,
    )
    from trellis2.representations import Mesh
    from trellis2.renderers import MeshRenderer

    report: Dict[str, Any] = {}
    report["gpu"] = torch.cuda.get_device_name(0)
    report["capability"] = list(torch.cuda.get_device_capability(0))
    report["total_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2)
    report["torch"] = torch.__version__
    report["cuda"] = torch.version.cuda
    report["torch_arch_list"] = torch.cuda.get_arch_list()
    report["flex_gemm"] = getattr(flex_gemm, "__version__", "unknown")
    report["o_voxel"] = getattr(o_voxel, "__version__", "unknown")
    report["utils3d"] = getattr(utils3d, "__version__", "unknown")
    print(json.dumps(report, indent=2), flush=True)

    if report["capability"][0] < 10:
        raise RuntimeError(f"Expected a Blackwell GPU, got compute capability {report['capability']}")
    if int(report["total_memory_gb"]) < 200:
        raise RuntimeError(f"Expected a B300 with 262 GB of HBM, got {report['total_memory_gb']} GB")

    torch.manual_seed(0)
    grid = torch.cartesian_prod(
        torch.arange(4, device="cuda"),
        torch.arange(4, device="cuda"),
        torch.arange(4, device="cuda"),
    )
    coords = torch.cat([torch.zeros_like(grid[:, :1]), grid], dim=-1).int()
    conv = sp.SparseConv3d(8, 16, 3).cuda()
    conv_feats = torch.randn(coords.shape[0], 8, device="cuda", requires_grad=True)
    conv_out = conv(sp.SparseTensor(conv_feats, coords))
    assert conv_out.feats.shape == (coords.shape[0], 16), conv_out.feats.shape
    assert torch.isfinite(conv_out.feats).all(), "flex_gemm produced non finite activations"
    conv_out.feats.square().mean().backward()
    assert conv_feats.grad is not None and torch.isfinite(conv_feats.grad).all(), "flex_gemm backward failed"
    report["sparse_conv"] = "ok"

    seq_lens = [5, 7, 11]
    num_heads, head_dim = 4, 16
    total_tokens = sum(seq_lens)
    qkv_feats = torch.randn(
        total_tokens, 3, num_heads, head_dim, device="cuda", dtype=torch.float32, requires_grad=True
    )
    varlen_qkv = sp.VarLenTensor(qkv_feats, sp.VarLenTensor.layout_from_seqlen(seq_lens))
    varlen_out = sparse_scaled_dot_product_attention(varlen_qkv)
    expected_chunks = []
    offset = 0
    for length in seq_lens:
        chunk = qkv_feats[offset:offset + length]
        expected_chunks.append(
            _reference_attention(chunk[:, 0].transpose(0, 1), chunk[:, 1].transpose(0, 1), chunk[:, 2].transpose(0, 1))
        )
        offset += length
    expected = torch.cat(expected_chunks, dim=0).transpose(0, 1)
    torch.testing.assert_close(varlen_out.feats, expected, atol=1e-5, rtol=1e-5)
    varlen_out.feats.square().mean().backward()
    assert qkv_feats.grad is not None and torch.isfinite(qkv_feats.grad).all(), "sparse attention backward failed"
    report["sparse_attention_fp32"] = "ok"

    qkv_bf16 = qkv_feats.detach().to(torch.bfloat16).requires_grad_(True)
    varlen_bf16 = sparse_scaled_dot_product_attention(
        sp.VarLenTensor(qkv_bf16, sp.VarLenTensor.layout_from_seqlen(seq_lens))
    )
    torch.testing.assert_close(varlen_bf16.feats.float(), expected, atol=2e-2, rtol=2e-2)
    report["sparse_attention_bf16"] = "ok"

    dense_qkv = torch.randn(2, 17, 3, num_heads, head_dim, device="cuda", dtype=torch.float32)
    dense_out = scaled_dot_product_attention(dense_qkv)
    dense_expected = torch.stack([
        _reference_attention(
            dense_qkv[batch, :, 0].transpose(0, 1),
            dense_qkv[batch, :, 1].transpose(0, 1),
            dense_qkv[batch, :, 2].transpose(0, 1),
        ).transpose(0, 1)
        for batch in range(dense_qkv.shape[0])
    ])
    torch.testing.assert_close(dense_out, dense_expected, atol=1e-5, rtol=1e-5)
    report["dense_attention"] = "ok"

    window_coords = []
    for batch in range(2):
        batch_coords = torch.cartesian_prod(
            torch.arange(4, device="cuda"),
            torch.arange(4, device="cuda"),
            torch.arange(4, device="cuda"),
        )
        window_coords.append(
            torch.cat([torch.full_like(batch_coords[:, :1], batch), batch_coords], dim=-1)
        )
    window_coords_tensor = torch.cat(window_coords, dim=0).int()
    window_feats = torch.randn(
        window_coords_tensor.shape[0], 3, num_heads, head_dim,
        device="cuda", dtype=torch.float32, requires_grad=True,
    )
    window_tensor = sp.SparseTensor(window_feats, window_coords_tensor)
    windowed_out = sparse_windowed_scaled_dot_product_self_attention(window_tensor, window_size=5)
    full_reference = []
    tokens_per_batch = 4 ** 3
    for batch in range(2):
        chunk = window_feats[batch * tokens_per_batch:(batch + 1) * tokens_per_batch]
        full_reference.append(
            _reference_attention(chunk[:, 0].transpose(0, 1), chunk[:, 1].transpose(0, 1), chunk[:, 2].transpose(0, 1))
        )
    expected_windowed = torch.stack(full_reference, dim=0).permute(0, 2, 1, 3).reshape(-1, num_heads, head_dim)
    torch.testing.assert_close(windowed_out.feats, expected_windowed, atol=1e-5, rtol=1e-5)
    windowed_small = sparse_windowed_scaled_dot_product_self_attention(
        sp.SparseTensor(window_feats, window_coords_tensor), window_size=2
    )
    assert torch.isfinite(windowed_small.feats).all(), "windowed attention produced non finite activations"
    windowed_small.feats.square().mean().backward()
    assert window_feats.grad is not None and torch.isfinite(window_feats.grad).all(), "windowed attention backward failed"
    report["windowed_attention"] = "ok"

    serialize_coords = torch.cartesian_prod(
        torch.arange(8, device="cuda"),
        torch.arange(8, device="cuda"),
        torch.arange(8, device="cuda"),
    ).int()
    codes = o_voxel.serialize.encode_seq(serialize_coords, mode="hilbert")
    decoded = o_voxel.serialize.decode_seq(codes, mode="hilbert")
    reencoded = o_voxel.serialize.encode_seq(decoded.int(), mode="hilbert")
    torch.testing.assert_close(torch.sort(codes).values, torch.sort(reencoded).values)
    vxz_path = Path("/tmp/verify_roundtrip.vxz")
    roundtrip_coords = serialize_coords.cpu()
    roundtrip_attrs = {
        "vertices": torch.randint(0, 255, (roundtrip_coords.shape[0], 3), dtype=torch.uint8),
        "intersected": torch.randint(0, 8, (roundtrip_coords.shape[0], 1), dtype=torch.uint8),
    }
    o_voxel.io.write(str(vxz_path), roundtrip_coords, roundtrip_attrs)
    read_coords, read_attrs = o_voxel.io.read_vxz(str(vxz_path), num_threads=4)
    assert read_coords.shape == roundtrip_coords.shape, (read_coords.shape, roundtrip_coords.shape)
    assert read_coords.dtype == roundtrip_coords.dtype, (read_coords.dtype, roundtrip_coords.dtype)
    torch.testing.assert_close(read_coords.int(), roundtrip_coords.int())
    for name, value in roundtrip_attrs.items():
        torch.testing.assert_close(read_attrs[name].int(), value.int())
    vxz_path.unlink()
    report["o_voxel"] = "ok"

    glctx = dr.RasterizeCudaContext()
    vertices = torch.tensor(
        [[[-0.5, -0.5, 0.0, 1.0], [0.5, -0.5, 0.0, 1.0], [0.0, 0.5, 0.0, 1.0]]],
        device="cuda", dtype=torch.float32,
    )
    faces = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
    rast, _ = dr.rasterize(glctx, vertices, faces, resolution=[64, 64])
    covered = int((rast[..., 3] > 0).sum().item())
    assert covered > 0, "nvdiffrast rasterised an empty image"
    report["nvdiffrast_pixels"] = covered

    base_cubemap = torch.full((6, 32, 32, 3), 0.5, device="cuda", dtype=torch.float32)
    env_light = EnvironmentLight(base_cubemap)
    env_light.build_mips()
    assert env_light.get_mip(1.0) is not None, "nvdiffrec failed to build the environment mip chain"
    cube_vertices = torch.tensor(
        [[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)],
        device="cuda", dtype=torch.float32,
    )
    cube_faces = torch.tensor(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
         [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]],
        device="cuda", dtype=torch.int32,
    )
    cu_mesh = cumesh.CuMesh()
    cu_mesh.init(cube_vertices.contiguous(), cube_faces.contiguous())
    assert cu_mesh.num_vertices == 8, cu_mesh.num_vertices
    assert cu_mesh.num_faces == 12, cu_mesh.num_faces
    report["cumesh"] = "ok"

    renderer = MeshRenderer({"near": 1.0, "far": 3.0})
    renderer.rendering_options.resolution = 128
    renderer.rendering_options.ssaa = 2
    mesh = Mesh(vertices=cube_vertices, faces=cube_faces.long()).cuda()
    camera_origin = torch.tensor([0.0, 0.0, 2.0], device="cuda", dtype=torch.float32)
    extrinsics = utils3d.torch.extrinsics_look_at(
        camera_origin,
        torch.tensor([0.0, 0.0, 0.0], device="cuda", dtype=torch.float32),
        torch.tensor([0.0, 0.0, 1.0], device="cuda", dtype=torch.float32),
    )
    fov = torch.deg2rad(torch.tensor(30.0, device="cuda", dtype=torch.float32))
    intrinsics = utils3d.torch.intrinsics_from_fov_xy(fov, fov)
    rendered = renderer.render(mesh, extrinsics, intrinsics)
    assert "normal" in rendered, sorted(rendered.keys())
    assert torch.isfinite(rendered["normal"]).all(), "renderer produced non finite output"
    assert float(rendered["mask"].sum()) > 0, "renderer produced an empty image"
    report["renderer"] = "ok"

    from trellis2.modules.sparse.attention import SparseMultiHeadAttention

    attention_channels, attention_heads = 32, 4
    module_coords = torch.cartesian_prod(
        torch.arange(4, device="cuda"),
        torch.arange(4, device="cuda"),
        torch.arange(2, device="cuda"),
    ).int()
    module_coords = torch.cat([torch.zeros_like(module_coords[:, :1]), module_coords], dim=-1)
    module_feats = torch.randn(module_coords.shape[0], attention_channels, device="cuda", requires_grad=True)
    self_attention = SparseMultiHeadAttention(
        attention_channels, attention_heads, use_rope=True, qk_rms_norm=True
    ).cuda()
    self_output = self_attention(sp.SparseTensor(module_feats, module_coords))
    assert self_output.feats.shape == module_feats.shape, self_output.feats.shape
    assert torch.isfinite(self_output.feats).all(), "self attention produced non finite activations"
    context_feats = torch.randn(module_coords.shape[0], attention_channels * 2, device="cuda")
    cross_attention = SparseMultiHeadAttention(
        attention_channels, attention_heads, ctx_channels=attention_channels * 2,
        type="cross", qk_rms_norm=True,
    ).cuda()
    cross_output = cross_attention(
        sp.SparseTensor(module_feats, module_coords),
        context=sp.VarLenTensor(context_feats, sp.VarLenTensor.layout_from_seqlen([module_coords.shape[0]])),
    )
    assert cross_output.feats.shape == module_feats.shape, cross_output.feats.shape
    assert torch.isfinite(cross_output.feats).all(), "cross attention produced non finite activations"
    (self_output.feats.square().mean() + cross_output.feats.square().mean()).backward()
    assert module_feats.grad is not None and torch.isfinite(module_feats.grad).all()
    report["attention_modules"] = "ok"

    from trellis2.models import SparseStructureFlowModel

    for pe_mode in ("ape", "rope"):
        flow_model = SparseStructureFlowModel(
            resolution=8,
            in_channels=4,
            out_channels=4,
            model_channels=32,
            cond_channels=16,
            num_blocks=2,
            num_heads=4,
            pe_mode=pe_mode,
        ).cuda()
        flow_x = torch.randn(2, 4, 8, 8, 8, device="cuda", requires_grad=True)
        flow_t = torch.rand(2, device="cuda")
        flow_cond = torch.randn(2, 12, 16, device="cuda")
        flow_out = flow_model(flow_x, flow_t, flow_cond)
        assert flow_out.shape == flow_x.shape, flow_out.shape
        assert torch.isfinite(flow_out).all(), f"flow model ({pe_mode}) produced non finite output"
        flow_out.square().mean().backward()
        assert flow_x.grad is not None and torch.isfinite(flow_x.grad).all()
        report[f"flow_model_{pe_mode}"] = "ok"

    from torch.utils.tensorboard import SummaryWriter
    from trellis2.datasets import FlexiDualGridDataset
    from trellis2.models import ElasticSLatFlowModel, FlexiDualGridVaeEncoder
    from transformers import DINOv3ViTModel

    writer = SummaryWriter("/tmp/verify_tb")
    writer.add_scalar("verify/ok", 1.0, 0)
    writer.close()
    report["tensorboard"] = "ok"
    report["transformers"] = TRANSFORMERS_VERSION
    report["imports"] = [
        FlexiDualGridDataset.__name__,
        ElasticSLatFlowModel.__name__,
        FlexiDualGridVaeEncoder.__name__,
        DINOv3ViTModel.__name__,
    ]

    print(json.dumps(report, indent=2), flush=True)
    print("[verify] every check passed", flush=True)
    return report


@app.function(
    image=training_image,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,
    cpu=8.0,
    memory=32 * 1024,
    secrets=_hf_secrets(),
    env=RUNTIME_ENV,
)
def prefetch_weights(
    repositories: Optional[Sequence[Sequence[Any]]] = None,
) -> Dict[str, str]:
    from huggingface_hub import snapshot_download

    volume.reload()
    selected = repositories or PRETRAINED_REPOSITORIES
    resolved: Dict[str, str] = {}
    for entry in selected:
        repo_id, allow_patterns = entry[0], list(entry[1])
        print(f"[modal] downloading {repo_id} ({', '.join(allow_patterns)})", flush=True)
        resolved[repo_id] = snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            max_workers=8,
        )
        print(f"[modal] {repo_id} -> {resolved[repo_id]}", flush=True)
    volume.commit()
    return resolved


@app.local_entrypoint()
def main(
    config: str = "configs/scvae/shape_vae_next_dc_f16c32_fp16.json",
    data_dir: str = "",
    run_name: str = "",
    overrides: str = "{}",
    ckpt: str = "latest",
    max_steps: int = 0,
    batch_size: int = 0,
    i_save: int = DEFAULT_I_SAVE,
    detach: bool = False,
    verify_first: bool = False,
    extra_train_args: str = "",
) -> None:
    if not data_dir:
        raise ValueError(
            "--data-dir is required. Pass either a directory or the JSON dataset specification "
            'documented in data_toolkit/README.md, e.g. \'{"MyDataset": {"base": "/vol/datasets/MyDataset"}}\''
        )

    parsed_overrides = json.loads(overrides)
    if not isinstance(parsed_overrides, dict):
        raise TypeError(f"--overrides must be a JSON object, got {type(parsed_overrides).__name__}")

    trainer_overrides: Dict[str, Any] = {}
    if max_steps > 0:
        trainer_overrides["max_steps"] = max_steps
    if batch_size > 0:
        trainer_overrides["batch_size_per_gpu"] = batch_size
    if i_save > 0:
        trainer_overrides["i_save"] = i_save
    if trainer_overrides:
        parsed_overrides = _deep_merge(parsed_overrides, {"trainer": {"args": trainer_overrides}})

    resolved_run_name = run_name or f"{Path(config).stem}-{time.strftime('%Y%m%d-%H%M%S')}"
    output_dir = f"{RESULTS_ROOT}/{resolved_run_name}"
    command_line_args = extra_train_args.split() if extra_train_args else []

    print(f"[modal] app             : {APP_NAME}")
    print(f"[modal] gpu             : {GPU_SPEC} x {GPU_COUNT}")
    print(f"[modal] config          : {config}")
    print(f"[modal] output_dir      : {output_dir}")
    print(f"[modal] overrides       : {json.dumps(parsed_overrides, indent=2)}")
    print(f"[modal] extra args      : {command_line_args}")

    if verify_first:
        verify.remote()

    if detach:
        call = train.spawn(config, data_dir, output_dir, parsed_overrides, ckpt, command_line_args)
        print(f"[modal] submitted function call {call.object_id}")
        print("[modal] follow the logs with: modal app logs " + APP_NAME)
        return

    result = train.remote(config, data_dir, output_dir, parsed_overrides, ckpt, command_line_args)
    print(json.dumps(result, indent=2))
