#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════
# AI Video Motion Control — Launcher
# Starts ComfyUI backend + Gradio frontend
# ═══════════════════════════════════════════════════════
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMFYUI_DIR="${PROJECT_DIR}/ComfyUI"
VENV_DIR="${PROJECT_DIR}/venv"

RED='\033[0;31m'
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# Check setup
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${RED}[✗] Virtual environment not found. Run ./setup.sh first.${NC}"
    exit 1
fi

source "${VENV_DIR}/bin/activate"

# Cleanup function
cleanup() {
    echo ""
    echo -e "${CYAN}[i] Shutting down...${NC}"
    if [ -n "${COMFYUI_PID:-}" ]; then
        kill "$COMFYUI_PID" 2>/dev/null || true
        wait "$COMFYUI_PID" 2>/dev/null || true
    fi
    echo -e "${GREEN}[✓] All processes stopped.${NC}"
}
trap cleanup EXIT INT TERM

# ── Start ComfyUI (headless) ─────────────────────────
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo -e "${CYAN}   AI Video Motion Control${NC}"
echo -e "${CYAN}═══════════════════════════════════════════${NC}"
echo ""

echo -e "${CYAN}[i] Starting ComfyUI backend (headless)...${NC}"
cd "$COMFYUI_DIR"
python main.py \
    --listen 127.0.0.1 \
    --port 8188 \
    --force-fp16 \
    --lowvram \
    --preview-method auto \
    --dont-print-server &
COMFYUI_PID=$!
cd "$PROJECT_DIR"

# Wait for ComfyUI to be ready
echo -e "${CYAN}[i] Waiting for ComfyUI to initialize...${NC}"
MAX_WAIT=120
WAITED=0
until curl -s http://127.0.0.1:8188/system_stats > /dev/null 2>&1; do
    sleep 2
    WAITED=$((WAITED + 2))
    if [ "$WAITED" -ge "$MAX_WAIT" ]; then
        echo -e "${RED}[✗] ComfyUI did not start within ${MAX_WAIT}s.${NC}"
        exit 1
    fi
done
echo -e "${GREEN}[✓] ComfyUI is ready (took ${WAITED}s).${NC}"

# ── Start Gradio frontend ────────────────────────────
echo -e "${CYAN}[i] Launching Gradio interface...${NC}"
echo ""
python "${PROJECT_DIR}/app.py"
