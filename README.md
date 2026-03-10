# wearit

Virtual try-on pipeline using CatVTON on a T4 GPU.

## Structure

```
wearit/
  setup.sh        install deps and download weights
  pipeline.py     CatVTON inference logic
  server.py       FastAPI server
  static/
    index.html    browser UI
```

## Setup

```bash
chmod +x setup.sh
./setup.sh
```

This installs all Python dependencies and downloads:
- CatVTON weights (~4GB) from HuggingFace
- SD inpainting base (~4GB) from HuggingFace

Requires ~8GB free disk space and ~8GB VRAM.

## Run

```bash
python server.py
```

Server starts at http://0.0.0.0:8000

Open in browser and upload a person photo + garment image.

## API

POST /tryon
  person   - image file (jpg/png)
  garment  - image file (jpg/png)
  category - upper | lower | full (default: upper)
  steps    - 5-50 (default: 20, higher = better quality but slower)

Returns JSON:
  { "image": "data:image/png;base64,..." }

GET /health
  Returns { "status": "ok" } when server is running.

## Notes

- First request loads the model into GPU memory (~20-30 seconds)
- Subsequent requests take 10-20 seconds on T4
- Lower steps (10-15) for faster results, higher (30-50) for better quality
- Category "upper" works best — most training data is upper body garments