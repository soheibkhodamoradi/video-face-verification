#!/usr/bin/env python3
"""Optional Gradio interface: run with `python app.py`."""

from __future__ import annotations

from functools import lru_cache

import gradio as gr

from face_verify.verify import FaceVerifier


@lru_cache(maxsize=1)
def get_verifier(every: int) -> FaceVerifier:
    """Reuse downloaded models between submissions."""

    return FaceVerifier(every=every)


def verify_videos(
    video1: str | None,
    video2: str | None,
    strategy: str,
    threshold: float,
    every: int,
) -> str:
    if not video1 or not video2:
        return "Please upload both videos."
    try:
        result = get_verifier(int(every)).verify(
            video1, video2, strategy=strategy, threshold=threshold
        )
    except (ValueError, OSError) as exc:
        return f"**Error:** {exc}"
    return (
        f"## {result.decision.capitalize()}\n\n"
        f"**Cosine similarity:** {result.similarity:.6f}  \n"
        f"**Threshold:** {result.threshold:.3f}  \n"
        f"**Usable frames:** {result.usable_frames_video1} and "
        f"{result.usable_frames_video2}"
    )


with gr.Blocks(title="Face Verification") as demo:
    gr.Markdown("# CPU Face Verification\nUpload two videos, each showing one main person.")
    with gr.Row():
        video1 = gr.Video(label="First video")
        video2 = gr.Video(label="Second video")
    with gr.Row():
        strategy = gr.Radio(["mean", "pairwise"], value="mean", label="Aggregation")
        threshold = gr.Slider(-1, 1, value=0.4, step=0.01, label="Decision threshold")
        every = gr.Slider(1, 30, value=5, step=1, label="Sample every Nth frame")
    output = gr.Markdown()
    gr.Button("Verify", variant="primary").click(
        verify_videos,
        inputs=[video1, video2, strategy, threshold, every],
        outputs=output,
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
