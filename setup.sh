#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
# AI Video Motion Control — Setup Script
# MacBook Air M2 (Apple Silicon) optimized
# ═══════════════════════════════════════════════════════
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMFYUI_DIR="${PROJECT_DIR}/ComfyUI"
VENV_DIR="${PROJECT_DIR}/venv"

log()   { echo -e "${GREEN}[✓]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
info()  { echo -e "${CYAN}[i]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; exit 1; }

# ── 1. Python venv ────────────────────────────────────
echo ""
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo -e "${CYAN}   AI Video Motion Control — Setup${NC}"
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo ""

if [ ! -d "$VENV_DIR" ]; then
    info "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    log "Virtual environment created."
else
    warn "Virtual environment already exists, skipping."
fi

source "${VENV_DIR}/bin/activate"
pip install --upgrade pip -q

# ── 2. Install wrapper dependencies ──────────────────
info "Installing Gradio wrapper dependencies..."
pip install -r "${PROJECT_DIR}/requirements.txt" -q
log "Wrapper dependencies installed."

# ── 3. Clone ComfyUI ─────────────────────────────────
if [ ! -d "$COMFYUI_DIR" ]; then
    info "Cloning ComfyUI..."
    git clone https://github.com/comfyanonymous/ComfyUI.git "$COMFYUI_DIR"
    log "ComfyUI cloned."
else
    warn "ComfyUI already exists, pulling latest..."
    cd "$COMFYUI_DIR" && git pull && cd "$PROJECT_DIR"
fi

# ── 4. Install ComfyUI dependencies ──────────────────
info "Installing ComfyUI dependencies..."
pip install -r "${COMFYUI_DIR}/requirements.txt" -q

# PyTorch nightly with MPS support (Apple Silicon)
info "Installing PyTorch with MPS support..."
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cpu -q

# onnxruntime with CoreML for DWPose GPU acceleration on Apple Silicon
info "Installing onnxruntime (CoreML backend for DWPose)..."
pip install onnxruntime -q
log "ComfyUI dependencies installed."

# ── 5. ComfyUI Custom Nodes ──────────────────────────
CUSTOM_NODES_DIR="${COMFYUI_DIR}/custom_nodes"

# AnimateDiff Evolved
if [ ! -d "${CUSTOM_NODES_DIR}/ComfyUI-AnimateDiff-Evolved" ]; then
    info "Installing AnimateDiff Evolved node..."
    git clone https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved.git \
        "${CUSTOM_NODES_DIR}/ComfyUI-AnimateDiff-Evolved"
    log "AnimateDiff Evolved installed."
else
    warn "AnimateDiff Evolved already exists."
fi

# ControlNet Aux Preprocessors
if [ ! -d "${CUSTOM_NODES_DIR}/comfyui_controlnet_aux" ]; then
    info "Installing ControlNet Aux preprocessors..."
    git clone https://github.com/Fannovel16/comfyui_controlnet_aux.git \
        "${CUSTOM_NODES_DIR}/comfyui_controlnet_aux"
    pip install -r "${CUSTOM_NODES_DIR}/comfyui_controlnet_aux/requirements.txt" -q 2>/dev/null || true
    log "ControlNet Aux installed."
else
    warn "ControlNet Aux already exists."
fi

# ComfyUI-VideoHelperSuite (for video loading/saving)
if [ ! -d "${CUSTOM_NODES_DIR}/ComfyUI-VideoHelperSuite" ]; then
    info "Installing Video Helper Suite..."
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        "${CUSTOM_NODES_DIR}/ComfyUI-VideoHelperSuite"
    pip install -r "${CUSTOM_NODES_DIR}/ComfyUI-VideoHelperSuite/requirements.txt" -q 2>/dev/null || true
    log "Video Helper Suite installed."
else
    warn "Video Helper Suite already exists."
fi

# ComfyUI-IPAdapter-Plus (for face/style consistency)
if [ ! -d "${CUSTOM_NODES_DIR}/ComfyUI_IPAdapter_plus" ]; then
    info "Installing IP-Adapter Plus node..."
    git clone https://github.com/cubiq/ComfyUI_IPAdapter_plus.git \
        "${CUSTOM_NODES_DIR}/ComfyUI_IPAdapter_plus"
    log "IP-Adapter Plus installed."
else
    warn "IP-Adapter Plus already exists."
fi

# ── 6. Download Models ────────────────────────────────
download_model() {
    local url="$1"
    local dest="$2"
    local name="$3"

    if [ -f "$dest" ]; then
        warn "${name} already downloaded."
        return
    fi

    info "Downloading ${name}..."
    mkdir -p "$(dirname "$dest")"
    curl -L --progress-bar -o "$dest" "$url"
    log "${name} downloaded."
}

MODELS_DIR="${COMFYUI_DIR}/models"

# SD 1.5 Checkpoint (Light Pruned)
download_model \
    "https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned.safetensors" \
    "${MODELS_DIR}/checkpoints/v1-5-pruned.safetensors" \
    "SD 1.5 Checkpoint (Pruned)"

# AnimateDiff Motion Module
download_model \
    "https://huggingface.co/guoyww/animatediff/resolve/main/mm_sd_v15_v2.ckpt" \
    "${MODELS_DIR}/animatediff_models/mm_sd_v15_v2.ckpt" \
    "AnimateDiff Motion Module v2"

# ControlNet OpenPose
download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11p_sd15_openpose.pth" \
    "${MODELS_DIR}/controlnet/control_v11p_sd15_openpose.pth" \
    "ControlNet OpenPose"

# ControlNet Depth
download_model \
    "https://huggingface.co/lllyasviel/ControlNet-v1-1/resolve/main/control_v11f1p_sd15_depth.pth" \
    "${MODELS_DIR}/controlnet/control_v11f1p_sd15_depth.pth" \
    "ControlNet Depth"

# IP-Adapter LIGHT (SD 1.5) — lightest possible preset
download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15_light.safetensors" \
    "${MODELS_DIR}/ipadapter/ip-adapter_sd15_light.safetensors" \
    "IP-Adapter LIGHT (SD 1.5)"

# CLIP Vision Encoder (required by IP-Adapter)
download_model \
    "https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/model.safetensors" \
    "${MODELS_DIR}/clip_vision/sd1.5_model.safetensors" \
    "CLIP Vision Encoder (SD 1.5)"

# LCM-LoRA (Latent Consistency Model — 4x faster rendering)
download_model \
    "https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors" \
    "${MODELS_DIR}/loras/lcm-lora-sdv1-5.safetensors" \
    "LCM-LoRA (SD 1.5)"

# ── 7. Create output directory ────────────────────────
mkdir -p "${PROJECT_DIR}/outputs"

# ── Done ──────────────────────────────────────────────
echo ""
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}   Setup Complete!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════${NC}"
echo ""
echo -e "  Run ${CYAN}./start.sh${NC} to launch the application."
echo ""
