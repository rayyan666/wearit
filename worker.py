import sys
import os
import io
import hashlib
import base64
import numpy as np
import torch
from PIL import Image, ImageFilter
from pathlib import Path
from celery import Celery

import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

REPO_DIR = Path(__file__).parent / "catvton_repo"
sys.path.insert(0, str(REPO_DIR))

from model.pipeline import CatVTONPipeline as _CatVTONPipeline
from model.cloth_masker import AutoMasker
from utils import resize_and_crop, resize_and_padding

WEIGHTS_DIR = Path(__file__).parent / "weights" / "catvton"
BASE_MODEL = "booksforcharlie/stable-diffusion-inpainting"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

WIDTH = 768
HEIGHT = 1024

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

app = Celery("wearit", broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_track_started=True,
    result_expires=3600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_pool="solo",
)

SIZE_SCALE = {
    "XS": 0.82,
    "S":  0.88,
    "M":  0.94,
    "L":  1.00,
    "XL": 1.06,
    "XXL": 1.12,
}


def apply_size_scaling(garment: Image.Image, size: str) -> Image.Image:
    scale = SIZE_SCALE.get(size.upper(), 1.0)
    if scale == 1.0:
        return garment
    w, h = garment.size
    new_w, new_h = int(w * scale), int(h * scale)
    resized = garment.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    canvas.paste(resized, ((w - new_w) // 2, (h - new_h) // 2))
    return canvas


def remove_background(image: Image.Image) -> Image.Image:
    try:
        from rembg import remove
        result = remove(image)
        background = Image.new("RGB", result.size, (255, 255, 255))
        background.paste(result, mask=result.split()[3] if result.mode == "RGBA" else None)
        return background
    except Exception:
        return image


def repaint(person: Image.Image, mask: Image.Image, result: Image.Image) -> Image.Image:
    _, h = result.size
    ksize = h // 50
    if ksize % 2 == 0:
        ksize += 1
    mask = mask.filter(ImageFilter.GaussianBlur(ksize))
    person_np = np.array(person.resize(result.size, Image.LANCZOS))
    result_np = np.array(result)
    mask_np = np.array(mask.resize(result.size, Image.LANCZOS).convert("L")) / 255.0
    mask_np = np.stack([mask_np] * 3, axis=-1)
    out = person_np * (1 - mask_np) + result_np * mask_np
    return Image.fromarray(out.astype(np.uint8))


def upscale(image: Image.Image) -> Image.Image:
    try:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        upsampler = RealESRGANer(
            scale=2,
            model_path=str(Path(__file__).parent / "weights/realesrgan/RealESRGAN_x2plus.pth"),
            model=model,
            tile=400,
            tile_pad=10,
            pre_pad=0,
            half=True if DEVICE == "cuda" else False,
            device=DEVICE,
        )
        img_np = np.array(image)[:, :, ::-1]
        out_np, _ = upsampler.enhance(img_np, outscale=2)
        return Image.fromarray(out_np[:, :, ::-1])
    except Exception:
        return image


def image_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_to_image(data: str) -> Image.Image:
    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def cache_key(person_b64: str, garment_b64: str, category: str,
              size: str, steps: int, lower_b64: str | None) -> str:
    parts = person_b64[:64] + garment_b64[:64] + category + size + str(steps)
    if lower_b64:
        parts += lower_b64[:64]
    return "wearit:result:" + hashlib.sha256(parts.encode()).hexdigest()


_pipeline_instance = None


def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = {
            "pipe": _CatVTONPipeline(
                base_ckpt=BASE_MODEL,
                attn_ckpt=str(WEIGHTS_DIR),
                attn_ckpt_version="mix",
                weight_dtype=DTYPE,
                device=DEVICE,
                skip_safety_check=True,
            ),
            "masker": AutoMasker(
                densepose_ckpt=str(WEIGHTS_DIR / "DensePose"),
                schp_ckpt=str(WEIGHTS_DIR / "SCHP"),
                device=DEVICE,
            ),
        }
    return _pipeline_instance


def run_single(pipe, masker, person, garment, category, steps, guidance_scale, generator):
    mask_result = masker(person, mask_type=category)
    mask = mask_result["mask"]
    results = pipe(
        image=person,
        condition_image=garment,
        mask=mask,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        height=HEIGHT,
        width=WIDTH,
        generator=generator,
    )
    return results[0], mask


@app.task(bind=True, name="tryon")
def tryon_task(
    self,
    person_b64: str,
    garment_b64: str,
    category: str = "upper",
    lower_b64: str = None,
    steps: int = 20,
    size: str = "M",
    seed: int = 42,
    use_upscale: bool = True,
):
    import redis as redis_lib
    r = redis_lib.from_url(REDIS_URL)
    key = cache_key(person_b64, garment_b64, category, size, steps, lower_b64)

    cached = r.get(key)
    if cached:
        return {"image": cached.decode(), "cached": True}

    self.update_state(state="PROGRESS", meta={"step": "loading model"})
    p = get_pipeline()
    pipe, masker = p["pipe"], p["masker"]

    self.update_state(state="PROGRESS", meta={"step": "preprocessing"})
    person_img = b64_to_image(person_b64)
    garment_img = b64_to_image(garment_b64)

    person = resize_and_crop(person_img, (WIDTH, HEIGHT))
    garment_clean = remove_background(garment_img)
    upper = resize_and_padding(apply_size_scaling(garment_clean, size), (WIDTH, HEIGHT))

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    self.update_state(state="PROGRESS", meta={"step": "running inference"})

    if category == "overall" and lower_b64:
        lower_img = b64_to_image(lower_b64)
        lower_clean = remove_background(lower_img)
        lower = resize_and_padding(apply_size_scaling(lower_clean, size), (WIDTH, HEIGHT))

        result_upper, upper_mask = run_single(pipe, masker, person, upper, "upper", steps, 2.5, generator)
        result_upper = repaint(person, upper_mask, result_upper)

        gen2 = torch.Generator(device=DEVICE).manual_seed(seed)
        result, lower_mask = run_single(pipe, masker, result_upper, lower, "lower", steps, 3.5, gen2)
        result = repaint(result_upper, lower_mask, result)

    elif category == "overall" and not lower_b64:
        # Use "overall" mask so the dress covers the full body
        result, mask = run_single(pipe, masker, person, upper, "overall", steps, 2.5, generator)
        result = repaint(person, mask, result)

    else:
        result, mask = run_single(pipe, masker, person, upper, category, steps, 2.5, generator)
        result = repaint(person, mask, result)

    if use_upscale:
        self.update_state(state="PROGRESS", meta={"step": "upscaling"})
        result = upscale(result)

    self.update_state(state="PROGRESS", meta={"step": "encoding result"})
    result_b64 = "data:image/png;base64," + image_to_b64(result)

    r.setex(key, 3600, result_b64)

    return {"image": result_b64, "cached": False}