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

# ── M2 Safe Mode Defaults (384x384 + LCM) ─────────────
DEFAULT_WIDTH = 384
DEFAULT_HEIGHT = 384
DEFAULT_FRAMES = 12  # ~1.5 seconds at 8fps (lighter)
MAX_FRAMES = 16  # ~2 seconds cap
DEFAULT_FPS = 8
DEFAULT_STEPS = 6  # LCM needs only 4-8 steps
DEFAULT_CFG = 1.8  # LCM works best at low CFG (1.5-2.5)
DEFAULT_MOTION_SCALE = 1.0
DEFAULT_DENOISE = 0.70  # For video-to-video

# ── LCM (Latent Consistency Model) ────────────────────
LCM_ENABLED = True
LCM_LORA = "lcm-lora-sdv1-5.safetensors"
LCM_LORA_STRENGTH = 1.0
LCM_SAMPLER = "lcm"
LCM_SCHEDULER = "sgm_uniform"

# ── Model Names (must match files in ComfyUI/models/) ─
SD_CHECKPOINT = "v1-5-pruned-emaonly.safetensors"
ANIMATEDIFF_MODEL = "mm_sd_v15_v2.ckpt"
CONTROLNET_OPENPOSE = "control_v11p_sd15_openpose.pth"
CONTROLNET_DEPTH = "control_v11f1p_sd15_depth.pth"
IPADAPTER_MODEL = "ip-adapter-plus_sd15.safetensors"
CLIP_VISION_MODEL = "sd1.5_model.safetensors"

# ── IP-Adapter Defaults (M2-safe) ─────────────────────
IPADAPTER_WEIGHT = 0.7
IPADAPTER_NOISE = 0.05

# ── Sampler Settings ──────────────────────────────────
DEFAULT_SAMPLER = LCM_SAMPLER if LCM_ENABLED else "euler_ancestral"
DEFAULT_SCHEDULER = LCM_SCHEDULER if LCM_ENABLED else "normal"
SEED = -1  # -1 = random

# ── Tiled VAE ─────────────────────────────────────────
USE_TILED_VAE = True  # Prevents OOM on 8GB shared memory
VAE_TILE_SIZE = 256
