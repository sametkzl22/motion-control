"""
AI Video Motion Control — Gradio Frontend
MacBook Air M2 optimized local video generation studio.
"""
import gradio as gr

import config
from comfyui_client import ComfyUIClient

client = ComfyUIClient()

# ── Custom CSS ─────────────────────────────────────────
CSS = """
:root {
    --primary: #0ea5e9;
    --primary-hover: #0284c7;
    --surface: #0f172a;
    --surface-light: #1e293b;
    --border: #334155;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --accent: #22d3ee;
    --success: #10b981;
    --warning: #f59e0b;
}

.gradio-container {
    max-width: 1200px !important;
    background: var(--surface) !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

.dark .gradio-container {
    background: var(--surface) !important;
}

#app-title {
    text-align: center;
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 50%, var(--success) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    margin-bottom: 0 !important;
}

#app-subtitle {
    text-align: center;
    color: var(--text-muted) !important;
    font-size: 0.95rem !important;
    margin-top: -8px !important;
}

.tab-nav button {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

.tab-nav button.selected {
    border-bottom-color: var(--primary) !important;
    color: var(--primary) !important;
}

footer { display: none !important; }

.status-ready { color: var(--success) !important; font-weight: 600; }
.status-error { color: #ef4444 !important; font-weight: 600; }

.generate-btn {
    background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
    border: none !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 12px 32px !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}

.generate-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(14, 165, 233, 0.3) !important;
}
"""


# ── Backend Status Check ───────────────────────────────
def check_status():
    if client.is_ready():
        return "🟢 ComfyUI Backend Ready"
    return "🔴 ComfyUI Backend Offline — Run ./start.sh"


# ── Text-to-Video Generator ───────────────────────────
def generate_t2v(prompt, negative_prompt, width, height, frames, steps, cfg, motion_scale, seed, progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Prompt cannot be empty!")

    if not client.is_ready():
        raise gr.Error("ComfyUI backend is not running. Start it with ./start.sh")

    progress(0, desc="🎬 Initializing generation...")

    def on_progress(value, maximum):
        progress(value / maximum, desc=f"🎨 Rendering step {value}/{maximum}")

    try:
        result_path = client.generate_text_to_video(
            progress_callback=on_progress,
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=int(width),
            height=int(height),
            frames=int(frames),
            steps=int(steps),
            cfg=float(cfg),
            motion_scale=float(motion_scale),
            seed=int(seed),
        )

        if result_path:
            progress(1.0, desc="✅ Video generated!")
            return result_path
        raise gr.Error("No video output was generated.")

    except TimeoutError:
        raise gr.Error("Generation timed out. Try fewer steps or frames.")
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


# ── Video-to-Video Generator ──────────────────────────
def generate_v2v(video, prompt, negative_prompt, controlnet_type, strength, denoise, frames, steps, cfg, motion_scale, seed, ref_image, ipa_weight, ipa_noise, progress=gr.Progress()):
    if not video:
        raise gr.Error("Please upload a reference video!")
    if not prompt.strip():
        raise gr.Error("Style prompt cannot be empty!")
    if not client.is_ready():
        raise gr.Error("ComfyUI backend is not running. Start it with ./start.sh")

    progress(0, desc="📤 Uploading reference video...")

    def on_progress(value, maximum):
        progress(value / maximum, desc=f"🎨 Rendering step {value}/{maximum}")

    try:
        result_path = client.generate_video_to_video(
            progress_callback=on_progress,
            video_path=video,
            prompt=prompt,
            negative_prompt=negative_prompt,
            controlnet_type=controlnet_type,
            strength=float(strength),
            denoise=float(denoise),
            frames=int(frames),
            steps=int(steps),
            cfg=float(cfg),
            motion_scale=float(motion_scale),
            seed=int(seed),
            image_path=ref_image if ref_image else None,
            ipadapter_weight=float(ipa_weight),
            ipadapter_noise=float(ipa_noise),
        )

        if result_path:
            progress(1.0, desc="✅ Video generated!")
            return result_path
        raise gr.Error("No video output was generated.")

    except TimeoutError:
        raise gr.Error("Generation timed out. Try fewer steps or frames.")
    except Exception as e:
        raise gr.Error(f"Generation failed: {str(e)}")


# ── Build UI ───────────────────────────────────────────
def create_app():
    with gr.Blocks(title="AI Video Motion Control") as app:

        # Header
        gr.Markdown("# 🎬 AI Video Motion Control", elem_id="app-title")
        gr.Markdown("MacBook Air M2 · AnimateDiff · ControlNet · IP-Adapter · ComfyUI", elem_id="app-subtitle")

        # Status bar
        status = gr.Markdown(value=check_status)

        with gr.Tabs():
            # ── Tab 1: Text to Video ──────────────────
            with gr.TabItem("✨ Text to Video", id="t2v"):
                with gr.Row():
                    with gr.Column(scale=1):
                        t2v_prompt = gr.Textbox(
                            label="Prompt",
                            placeholder="A cyberpunk samurai walking through neon-lit streets, cinematic, 4k...",
                            lines=3,
                        )
                        t2v_neg = gr.Textbox(
                            label="Negative Prompt",
                            value="bad quality, blurry, distorted, watermark, text, deformed, ugly",
                            lines=2,
                        )

                        with gr.Row():
                            t2v_width = gr.Slider(256, 768, value=config.DEFAULT_WIDTH, step=64, label="Width")
                            t2v_height = gr.Slider(256, 768, value=config.DEFAULT_HEIGHT, step=64, label="Height")

                        with gr.Row():
                            t2v_frames = gr.Slider(8, config.MAX_FRAMES, value=config.DEFAULT_FRAMES, step=1, label="Frames")
                            t2v_steps = gr.Slider(10, 40, value=config.DEFAULT_STEPS, step=1, label="Steps")

                        with gr.Row():
                            t2v_cfg = gr.Slider(1.0, 15.0, value=config.DEFAULT_CFG, step=0.5, label="CFG Scale")
                            t2v_motion = gr.Slider(0.5, 1.5, value=config.DEFAULT_MOTION_SCALE, step=0.05, label="Motion Scale")

                        t2v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

                        t2v_btn = gr.Button("🚀 Generate Video", variant="primary", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        t2v_output = gr.Video(label="Generated Video", interactive=False)

                t2v_btn.click(
                    fn=generate_t2v,
                    inputs=[t2v_prompt, t2v_neg, t2v_width, t2v_height, t2v_frames, t2v_steps, t2v_cfg, t2v_motion, t2v_seed],
                    outputs=t2v_output,
                )

            # ── Tab 2: Video to Video ─────────────────
            with gr.TabItem("🔄 Motion Transfer", id="v2v"):
                with gr.Row():
                    with gr.Column(scale=1):
                        v2v_video = gr.Video(label="Reference Video (source motion)", sources=["upload"])
                        v2v_prompt = gr.Textbox(
                            label="Style Prompt",
                            placeholder="An anime character performing the same motion, vibrant colors, studio ghibli style...",
                            lines=3,
                        )
                        v2v_neg = gr.Textbox(
                            label="Negative Prompt",
                            value="bad quality, blurry, distorted, watermark, text, deformed, ugly",
                            lines=2,
                        )

                        v2v_controlnet = gr.Radio(
                            choices=["openpose", "depth"],
                            value="openpose",
                            label="ControlNet Type",
                            info="OpenPose = skeleton tracking · Depth = depth map",
                        )

                        gr.Markdown("### 🖼️ Face & Style Reference (IP-Adapter)")
                        v2v_ref_image = gr.Image(
                            label="Reference Person / Style",
                            type="filepath",
                            sources=["upload"],
                        )

                        with gr.Row():
                            v2v_ipa_weight = gr.Slider(0.3, 1.0, value=config.IPADAPTER_WEIGHT, step=0.05, label="IP-Adapter Weight", info="Higher = more faithful to reference")
                            v2v_ipa_noise = gr.Slider(0.0, 0.2, value=config.IPADAPTER_NOISE, step=0.01, label="IP-Adapter Noise", info="Low noise is safer for M2")

                        with gr.Row():
                            v2v_strength = gr.Slider(0.3, 1.0, value=0.85, step=0.05, label="ControlNet Strength")
                            v2v_denoise = gr.Slider(0.3, 1.0, value=config.DEFAULT_DENOISE, step=0.05, label="Denoise Strength")

                        with gr.Row():
                            v2v_frames = gr.Slider(8, config.MAX_FRAMES, value=config.DEFAULT_FRAMES, step=1, label="Frames")
                            v2v_steps = gr.Slider(10, 40, value=config.DEFAULT_STEPS, step=1, label="Steps")

                        with gr.Row():
                            v2v_cfg = gr.Slider(1.0, 15.0, value=config.DEFAULT_CFG, step=0.5, label="CFG Scale")
                            v2v_motion = gr.Slider(0.5, 1.5, value=config.DEFAULT_MOTION_SCALE, step=0.05, label="Motion Scale")

                        v2v_seed = gr.Number(label="Seed (-1 = random)", value=-1, precision=0)

                        v2v_btn = gr.Button("🔄 Transfer Motion", variant="primary", elem_classes=["generate-btn"])

                    with gr.Column(scale=1):
                        v2v_output = gr.Video(label="Generated Video", interactive=False)

                v2v_btn.click(
                    fn=generate_v2v,
                    inputs=[v2v_video, v2v_prompt, v2v_neg, v2v_controlnet, v2v_strength, v2v_denoise, v2v_frames, v2v_steps, v2v_cfg, v2v_motion, v2v_seed, v2v_ref_image, v2v_ipa_weight, v2v_ipa_noise],
                    outputs=v2v_output,
                )

        # Refresh status periodically
        timer = gr.Timer(10)
        timer.tick(fn=check_status, outputs=status)

    return app


# ── Entry Point ────────────────────────────────────────
if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True,
        show_error=True,
        css=CSS,
        theme=gr.themes.Base(primary_hue="sky", neutral_hue="slate"),
    )
