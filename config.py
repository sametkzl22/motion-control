"""
AI Video Motion Control — Configuration
MacBook Air M2 (Apple Silicon) optimized settings.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()
COMFYUI_DIR = BASE_DIR / "ComfyUI"
WORKFLOWS_DIR = BASE_DIR / "workflows"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── ComfyUI Server ────────────────────────────────────
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8188
COMFYUI_URL = f"http://{COMFYUI_HOST}:{COMFYUI_PORT}"

# ── M2 / MPS Optimization ─────────────────────────────
DEVICE = "mps"  # Metal Performance Shaders
DTYPE = "fp16"  # Half-precision — saves ~50% VRAM
LOW_VRAM = True  # Aggressive memory management
FORCE_FP16 = True  # Force fp16 on all models

# ComfyUI launch flags for M2
COMFYUI_EXTRA_ARGS = [
    "--force-fp16",
    "--preview-method", "auto",
]
if LOW_VRAM:
    COMFYUI_EXTRA_ARGS.append("--lowvram")

# ── Generation Defaults ───────────────────────────────
DEFAULT_WIDTH = 512
DEFAULT_HEIGHT = 512
DEFAULT_FRAMES = 16  # ~2 seconds at 8fps
MAX_FRAMES = 24  # ~3 seconds at 8fps
DEFAULT_FPS = 8
DEFAULT_STEPS = 20
DEFAULT_CFG = 7.5
DEFAULT_MOTION_SCALE = 1.0
DEFAULT_DENOISE = 0.75  # For video-to-video

# ── Model Names (must match files in ComfyUI/models/) ─
SD_CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
ANIMATEDIFF_MODEL = "mm_sd_v15_v2.ckpt"
CONTROLNET_OPENPOSE = "control_v11p_sd15_openpose.pth"
CONTROLNET_DEPTH = "control_v11f1p_sd15_depth.pth"
IPADAPTER_MODEL = "ip-adapter-plus_sd15.safetensors"
CLIP_VISION_MODEL = "sd1.5_model.safetensors"

# ── IP-Adapter Defaults (M2-safe) ─────────────────────
IPADAPTER_WEIGHT = 0.7   # 0.6-0.8 range balances fidelity vs. creativity
IPADAPTER_NOISE = 0.05   # Low noise keeps M2 memory stable

# ── Sampler Settings ──────────────────────────────────
DEFAULT_SAMPLER = "euler_ancestral"
DEFAULT_SCHEDULER = "normal"
SEED = -1  # -1 = random
