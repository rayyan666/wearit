#!/bin/bash

set -e

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers==0.27.2 transformers accelerate
pip install fastapi uvicorn python-multipart pillow huggingface_hub

python - <<'EOF'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="zhengchong/CatVTON",
    local_dir="./weights/catvton",
    ignore_patterns=["*.msgpack", "*.h5"]
)

snapshot_download(
    repo_id="runwayml/stable-diffusion-inpainting",
    local_dir="./weights/sd-inpainting",
    ignore_patterns=["*.msgpack", "*.h5"]
)

print("weights downloaded")
EOF

echo "setup complete"