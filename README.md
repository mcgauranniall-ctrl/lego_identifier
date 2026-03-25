---
title: LEGO Brick Identifier
emoji: 🧱
colorFrom: red
colorTo: yellow
sdk: gradio
sdk_version: "5.0.0"
python_version: "3.11"
app_file: app.py
pinned: false
---

# LEGO Brick Identifier

Upload a photo of LEGO pieces. SAM2 detects each brick, then the [Brickognize API](https://brickognize.com/) identifies the part number and links to BrickLink.

## How it works

1. **SAM2** (Segment Anything Model 2) segments every object in your photo
2. Each cropped segment is sent to the **Brickognize API** for part identification
3. Results show part IDs, names, confidence scores, and BrickLink links
