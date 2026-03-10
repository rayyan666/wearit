import sys
import os
import numpy as np
import torch
from PIL import Image
from pathlib import Path

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

# Scale factors per clothing size — adjusts how the garment fills the frame
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
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = garment.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    canvas.paste(resized, (paste_x, paste_y))
    return canvas


def blend_results(base: Image.Image, overlay: Image.Image, mask: Image.Image) -> Image.Image:
    base_np = np.array(base).astype(np.float32)
    overlay_np = np.array(overlay).astype(np.float32)
    mask_np = np.array(mask.convert("L")).astype(np.float32) / 255.0
    mask_np = np.stack([mask_np] * 3, axis=-1)
    blended = base_np * (1 - mask_np) + overlay_np * mask_np
    return Image.fromarray(blended.astype(np.uint8))


class WearItPipeline:
    def __init__(self):
        self.pipe = _CatVTONPipeline(
            base_ckpt=BASE_MODEL,
            attn_ckpt=str(WEIGHTS_DIR),
            attn_ckpt_version="mix",
            weight_dtype=DTYPE,
            device=DEVICE,
            skip_safety_check=True,
        )

        self.masker = AutoMasker(
            densepose_ckpt=str(WEIGHTS_DIR / "DensePose"),
            schp_ckpt=str(WEIGHTS_DIR / "SCHP"),
            device=DEVICE,
        )

    def _run_single(
        self,
        person: Image.Image,
        garment: Image.Image,
        category: str,
        steps: int,
        guidance_scale: float,
        generator: torch.Generator,
    ) -> tuple[Image.Image, Image.Image]:
        mask_result = self.masker(person, mask_type=category)
        mask = mask_result["mask"]
        results = self.pipe(
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

    def run(
        self,
        person_image: Image.Image,
        upper_garment: Image.Image,
        category: str = "upper",
        lower_garment: Image.Image = None,
        steps: int = 20,
        guidance_scale: float = 2.5,
        seed: int = 42,
        size: str = "M",
    ) -> Image.Image:
        person = resize_and_crop(person_image.convert("RGB"), (WIDTH, HEIGHT))
        upper = resize_and_padding(apply_size_scaling(upper_garment.convert("RGB"), size), (WIDTH, HEIGHT))

        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        if category == "overall" and lower_garment is not None:
            # Pass 1: apply upper garment
            result_upper, upper_mask = self._run_single(person, upper, "upper", steps, guidance_scale, generator)

            # Pass 2: apply lower garment on top of pass 1 result
            lower = resize_and_padding(apply_size_scaling(lower_garment.convert("RGB"), size), (WIDTH, HEIGHT))
            generator2 = torch.Generator(device=DEVICE).manual_seed(seed)
            result_final, _ = self._run_single(result_upper, lower, "lower", steps, guidance_scale, generator2)
            return result_final

        elif category == "overall" and lower_garment is None:
            # Only replace upper, keep lower from original person
            result, _ = self._run_single(person, upper, "upper", steps, guidance_scale, generator)
            return result

        else:
            result, _ = self._run_single(person, upper, category, steps, guidance_scale, generator)
            return result


_pipeline: WearItPipeline = None


def get_pipeline() -> WearItPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = WearItPipeline()
    return _pipeline