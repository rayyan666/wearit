#!/bin/bash
set -e

echo "installing production dependencies..."

pip install rembg[gpu]
pip install celery redis
pip install basicsr facexlib gfpgan
pip install realesrgan

echo "downloading Real-ESRGAN weights..."
mkdir -p weights/realesrgan
wget -q -O weights/realesrgan/RealESRGAN_x2plus.pth \
  https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth

echo "checking redis..."
if ! command -v redis-server &> /dev/null; then
  sudo yum install -y redis || sudo apt-get install -y redis-server
fi

echo "starting redis..."
redis-server --daemonize yes --port 6379

echo "production setup complete"
echo ""
echo "to start:"
echo "  terminal 1: python server.py"
echo "  terminal 2: celery -A worker worker --loglevel=info --concurrency=1"