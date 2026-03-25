"""
Brickognize API client for LEGO part identification.

Sends cropped LEGO piece images to the Brickognize API and returns
part identification results with BrickLink IDs and confidence scores.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import requests
from PIL import Image

API_URL = "https://api.brickognize.com/predict/"
TIMEOUT = 15  # seconds per request


@dataclass
class BrickResult:
    """A single identification result from Brickognize."""

    part_id: str
    name: str
    score: float
    image_url: str
    bricklink_url: str


def identify(image: Image.Image) -> list[BrickResult]:
    """
    Send an image to Brickognize and return identification results.

    Parameters
    ----------
    image : PIL.Image
        Cropped LEGO piece image (RGB).

    Returns
    -------
    list[BrickResult]
        Ranked results from most to least confident.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    resp = requests.post(
        API_URL,
        headers={"accept": "application/json"},
        files={"query_image": ("crop.png", buf, "image/png")},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    results = []
    for item in data.get("items", []):
        part_id = item.get("id", "")
        name = item.get("name", "")
        score = float(item.get("score", 0))
        img_url = item.get("img_url", "")

        bl_url = ""
        if part_id:
            bl_url = f"https://www.bricklink.com/v2/catalog/catalogitem.page?P={part_id}"

        results.append(BrickResult(
            part_id=part_id,
            name=name,
            score=score,
            image_url=img_url,
            bricklink_url=bl_url,
        ))

    return results
