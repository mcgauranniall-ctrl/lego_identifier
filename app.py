"""
Gradio UI for LEGO Brick Identifier — runs on Hugging Face Spaces.

Upload a photo of LEGO pieces, get part IDs and BrickLink links.
Uses SAM2 for detection + Brickognize API for identification.
"""

import sys
from pathlib import Path

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from brickognize.pipeline import analyze_image


def draw_detections(image: Image.Image, result) -> Image.Image:
    """Draw bounding boxes and labels on the image."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except (OSError, IOError):
        font = ImageFont.load_default()

    colors = ["#FF4444", "#44FF44", "#4444FF", "#FFAA00", "#FF44FF",
              "#44FFFF", "#FF8844", "#88FF44", "#4488FF", "#FF4488"]

    for i, det in enumerate(result.detections):
        color = colors[i % len(colors)]
        x1, y1, x2, y2 = det.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        if det.results:
            top = det.results[0]
            label = f"{top.part_id} {top.name} ({top.score:.0%})"
            bbox = draw.textbbox((x1, y1 - 18), label, font=font)
            draw.rectangle(bbox, fill=color)
            draw.text((x1, y1 - 18), label, fill="white", font=font)

    return annotated


def identify_bricks(image: Image.Image) -> tuple[Image.Image, str]:
    """Run the full pipeline and return annotated image + results text."""
    if image is None:
        return None, "Please upload an image."

    result = analyze_image(image, top_k=5)

    # Build results text
    lines = []
    if result.grouped_parts:
        lines.append(f"**Found {result.total_pieces} piece(s), "
                     f"{result.unique_parts} unique part(s)**\n")
        for g in result.grouped_parts:
            lines.append(f"- **{g.count}x {g.name}** (ID: {g.part_id}, "
                         f"confidence: {g.best_score:.0%})")
            lines.append(f"  [BrickLink]({g.bricklink_url})")
    else:
        lines.append("No LEGO pieces identified.")

    lines.append(f"\n*Processing time: {result.processing_time_ms / 1000:.1f}s*")

    # Draw boxes on image
    annotated = draw_detections(image, result)

    return annotated, "\n".join(lines)


demo = gr.Interface(
    fn=identify_bricks,
    inputs=gr.Image(type="pil", label="Upload a photo of LEGO pieces"),
    outputs=[
        gr.Image(type="pil", label="Detected pieces"),
        gr.Markdown(label="Identification results"),
    ],
    title="LEGO Brick Identifier",
    description="Upload a photo of LEGO pieces. SAM2 detects each brick, "
                "then the Brickognize API identifies the part number.",
    examples=[],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
