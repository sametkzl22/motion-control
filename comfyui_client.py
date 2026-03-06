"""
AI Video Motion Control — ComfyUI API Client
Handles communication with ComfyUI backend via REST + WebSocket.
"""
import json
import uuid
import time
import shutil
from pathlib import Path
from typing import Optional

import requests
import websocket
from tqdm import tqdm

import config


class ComfyUIClient:
    """Wrapper for ComfyUI's REST and WebSocket API."""

    def __init__(self, host: str = config.COMFYUI_HOST, port: int = config.COMFYUI_PORT):
        self.base_url = f"http://{host}:{port}"
        self.ws_url = f"ws://{host}:{port}/ws"
        self.client_id = str(uuid.uuid4())

    # ── Health ─────────────────────────────────────────

    def is_ready(self) -> bool:
        """Check if ComfyUI server is responsive."""
        try:
            r = requests.get(f"{self.base_url}/system_stats", timeout=3)
            return r.status_code == 200
        except requests.ConnectionError:
            return False

    def wait_until_ready(self, timeout: int = 120):
        """Block until ComfyUI is responsive."""
        start = time.time()
        while time.time() - start < timeout:
            if self.is_ready():
                return True
            time.sleep(2)
        raise TimeoutError(f"ComfyUI did not respond within {timeout}s")

    # ── Prompt Execution ───────────────────────────────

    def queue_prompt(self, workflow: dict) -> str:
        """Send a workflow to ComfyUI and return the prompt_id."""
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        r = requests.post(f"{self.base_url}/prompt", json=payload)
        if r.status_code != 200:
            try:
                error_data = r.json()
                if "node_errors" in error_data:
                    errors = []
                    for nid, nerr in error_data["node_errors"].items():
                        for e in nerr.get("errors", []):
                            errors.append(f"Node #{nid} ({nerr.get('class_type','?')}): {e.get('message','unknown')}")
                    if errors:
                        raise RuntimeError("ComfyUI validation errors:\n" + "\n".join(errors))
                if "error" in error_data:
                    err = error_data["error"]
                    raise RuntimeError(f"ComfyUI error: {err.get('message', str(err))}")
            except (ValueError, KeyError):
                pass
            r.raise_for_status()
        return r.json()["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        """Retrieve execution history for a given prompt."""
        r = requests.get(f"{self.base_url}/history/{prompt_id}")
        r.raise_for_status()
        return r.json()

    def get_output_file(self, filename: str, subfolder: str = "", folder_type: str = "output") -> bytes:
        """Download a generated file from ComfyUI."""
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        r = requests.get(f"{self.base_url}/view", params=params)
        r.raise_for_status()
        return r.content

    def free_memory(self):
        """Flush ComfyUI VRAM + MPS cache after generation to prevent overheating."""
        try:
            requests.post(f"{self.base_url}/free", json={"unload_models": True, "free_memory": True}, timeout=5)
        except Exception:
            pass
        try:
            import torch
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    # ── WebSocket Progress Tracking ────────────────────

    def execute_and_wait(self, workflow: dict, progress_callback=None) -> dict:
        """
        Queue a workflow, track progress via WebSocket, return outputs.
        progress_callback(value, max) is called on each progress update.
        """
        prompt_id = self.queue_prompt(workflow)

        ws = websocket.WebSocket()
        ws.connect(f"{self.ws_url}?clientId={self.client_id}")

        try:
            while True:
                msg = ws.recv()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    msg_type = data.get("type")

                    if msg_type == "progress" and progress_callback:
                        d = data["data"]
                        progress_callback(d["value"], d["max"])

                    elif msg_type == "executing":
                        d = data["data"]
                        if d.get("node") is None and d.get("prompt_id") == prompt_id:
                            break  # Execution complete
        finally:
            ws.close()

        return self._collect_outputs(prompt_id)

    def _collect_outputs(self, prompt_id: str) -> dict:
        """Gather all output files from a completed prompt."""
        history = self.get_history(prompt_id)
        if prompt_id not in history:
            return {}

        outputs = history[prompt_id].get("outputs", {})
        result = {"videos": [], "images": []}

        for node_id, node_output in outputs.items():
            # Video outputs (VHS_VideoCombine)
            if "gifs" in node_output:
                for video_info in node_output["gifs"]:
                    result["videos"].append(video_info)

            # Image outputs
            if "images" in node_output:
                for img_info in node_output["images"]:
                    result["images"].append(img_info)

        return result

    # ── File Upload ────────────────────────────────────

    def upload_video(self, video_path: str) -> str:
        """Upload a video to ComfyUI's input directory. Returns the server filename."""
        path = Path(video_path)
        with open(path, "rb") as f:
            files = {"image": (path.name, f, "video/mp4")}
            data = {"overwrite": "true", "type": "input"}
            r = requests.post(f"{self.base_url}/upload/image", files=files, data=data)
            r.raise_for_status()
            return r.json()["name"]

    def upload_image(self, image_path: str) -> str:
        """Upload an image to ComfyUI's input directory. Returns the server filename."""
        path = Path(image_path)
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        with open(path, "rb") as f:
            files = {"image": (path.name, f, mime)}
            data = {"overwrite": "true", "type": "input"}
            r = requests.post(f"{self.base_url}/upload/image", files=files, data=data)
            r.raise_for_status()
            return r.json()["name"]

    # ── Workflow Builders ──────────────────────────────

    def build_text_to_video(
        self,
        prompt: str,
        negative_prompt: str = "bad quality, blurry, distorted, watermark, text",
        width: int = config.DEFAULT_WIDTH,
        height: int = config.DEFAULT_HEIGHT,
        frames: int = config.DEFAULT_FRAMES,
        steps: int = config.DEFAULT_STEPS,
        cfg: float = config.DEFAULT_CFG,
        motion_scale: float = config.DEFAULT_MOTION_SCALE,
        seed: int = config.SEED,
    ) -> dict:
        """Load and parameterize the text-to-video workflow."""
        workflow = self._load_workflow("text_to_video.json")

        if seed == -1:
            seed = int(time.time()) % (2**32)

        # Patch nodes
        workflow["2"]["inputs"]["text"] = prompt
        workflow["3"]["inputs"]["text"] = negative_prompt
        workflow["4"]["inputs"]["width"] = width
        workflow["4"]["inputs"]["height"] = height
        workflow["4"]["inputs"]["batch_size"] = frames
        workflow["5"]["inputs"]["motion_scale"] = motion_scale
        workflow["6"]["inputs"]["context_length"] = frames
        workflow["7"]["inputs"]["seed"] = seed
        workflow["7"]["inputs"]["steps"] = steps
        workflow["7"]["inputs"]["cfg"] = cfg
        workflow["7"]["inputs"]["sampler_name"] = config.DEFAULT_SAMPLER
        workflow["7"]["inputs"]["scheduler"] = config.DEFAULT_SCHEDULER
        workflow["9"]["inputs"]["frame_rate"] = config.DEFAULT_FPS

        # Tiled VAE tile size
        if config.USE_TILED_VAE and "8" in workflow:
            workflow["8"]["inputs"]["tile_size"] = config.VAE_TILE_SIZE

        # LCM LoRA strength
        if config.LCM_ENABLED and "15" in workflow:
            workflow["15"]["inputs"]["strength_model"] = config.LCM_LORA_STRENGTH

        return workflow

    def build_video_to_video(
        self,
        video_path: str,
        prompt: str,
        negative_prompt: str = "bad quality, blurry, distorted, watermark, text",
        controlnet_type: str = "openpose",
        strength: float = 0.85,
        denoise: float = config.DEFAULT_DENOISE,
        frames: int = config.DEFAULT_FRAMES,
        steps: int = config.DEFAULT_STEPS,
        cfg: float = config.DEFAULT_CFG,
        motion_scale: float = config.DEFAULT_MOTION_SCALE,
        seed: int = config.SEED,
        image_path: Optional[str] = None,
        ipadapter_weight: float = config.IPADAPTER_WEIGHT,
        ipadapter_noise: float = config.IPADAPTER_NOISE,
    ) -> dict:
        """Load and parameterize the video-to-video workflow."""
        # Upload video to ComfyUI
        server_filename = self.upload_video(video_path)

        workflow = self._load_workflow("video_to_video.json")

        if seed == -1:
            seed = int(time.time()) % (2**32)

        # ControlNet mapping
        cn_map = {
            "openpose": {
                "model": config.CONTROLNET_OPENPOSE,
                "preprocessor": "DWPreprocessor",
            },
            "depth": {
                "model": config.CONTROLNET_DEPTH,
                "preprocessor": "DepthAnythingPreprocessor",
            },
            "softedge": {
                "model": config.CONTROLNET_SOFTEDGE,
                "preprocessor": "PiDiNetPreprocessor",
            },
        }
        cn = cn_map.get(controlnet_type, cn_map["openpose"])

        # Patch core nodes
        workflow["2"]["inputs"]["text"] = prompt
        workflow["3"]["inputs"]["text"] = negative_prompt
        workflow["10"]["inputs"]["video"] = server_filename
        workflow["10"]["inputs"]["frame_load_cap"] = frames
        workflow["11"]["inputs"]["preprocessor"] = cn["preprocessor"]
        workflow["11"]["inputs"]["resolution"] = config.DEFAULT_WIDTH
        workflow["12"]["inputs"]["control_net_name"] = cn["model"]
        workflow["13"]["inputs"]["strength"] = strength
        workflow["5"]["inputs"]["motion_scale"] = motion_scale
        workflow["6"]["inputs"]["context_length"] = frames
        workflow["7"]["inputs"]["seed"] = seed
        workflow["7"]["inputs"]["steps"] = steps
        workflow["7"]["inputs"]["cfg"] = cfg
        workflow["7"]["inputs"]["denoise"] = denoise
        workflow["7"]["inputs"]["sampler_name"] = config.DEFAULT_SAMPLER
        workflow["7"]["inputs"]["scheduler"] = config.DEFAULT_SCHEDULER
        workflow["9"]["inputs"]["frame_rate"] = config.DEFAULT_FPS

        # Tiled VAE tile size
        if config.USE_TILED_VAE and "8" in workflow:
            workflow["8"]["inputs"]["tile_size"] = config.VAE_TILE_SIZE

        # LCM LoRA strength
        if config.LCM_ENABLED and "15" in workflow:
            workflow["15"]["inputs"]["strength_model"] = config.LCM_LORA_STRENGTH

        # IP-Adapter: patch if reference image provided, else strip nodes
        if image_path:
            server_image = self.upload_image(image_path)
            workflow["22"]["inputs"]["image"] = server_image
            workflow["23"]["inputs"]["weight"] = ipadapter_weight
        else:
            # Remove IP-Adapter nodes, connect model directly to AnimateDiff via LoRA
            for nid in ["20", "22", "23"]:
                workflow.pop(nid, None)
            workflow["5"]["inputs"]["model"] = ["15", 0]

        return workflow

    def _load_workflow(self, name: str) -> dict:
        """Load a workflow JSON from the workflows directory."""
        path = config.WORKFLOWS_DIR / name
        with open(path, "r") as f:
            return json.load(f)

    # ── High-Level Generate ────────────────────────────

    def generate_text_to_video(self, progress_callback=None, **kwargs) -> Optional[str]:
        """Full text-to-video pipeline: build → execute → save → cleanup."""
        workflow = self.build_text_to_video(**kwargs)
        outputs = self.execute_and_wait(workflow, progress_callback)
        result = self._save_first_video(outputs, "t2v")
        self.free_memory()
        return result

    def generate_video_to_video(self, progress_callback=None, **kwargs) -> Optional[str]:
        """Full video-to-video pipeline: build → execute → save → cleanup."""
        workflow = self.build_video_to_video(**kwargs)
        outputs = self.execute_and_wait(workflow, progress_callback)
        result = self._save_first_video(outputs, "v2v")
        self.free_memory()
        return result

    def _save_first_video(self, outputs: dict, prefix: str) -> Optional[str]:
        """Download the first video output and save locally."""
        if not outputs.get("videos"):
            return None

        video_info = outputs["videos"][0]
        filename = video_info["filename"]
        subfolder = video_info.get("subfolder", "")

        video_bytes = self.get_output_file(filename, subfolder)

        timestamp = int(time.time())
        out_name = f"{prefix}_{timestamp}.mp4"
        out_path = config.OUTPUT_DIR / out_name

        with open(out_path, "wb") as f:
            f.write(video_bytes)

        return str(out_path)
