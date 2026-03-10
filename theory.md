# wearit

**Virtual Try-On System**
*Technical Reference and Architecture Documentation*

---

## 1. Overview

wearit is a virtual garment try-on system that enables users to preview how clothing items will look on a person photograph without physically wearing them. The system accepts an image of a person and one or more garment images, processes them through a diffusion-based inpainting pipeline, and returns a photorealistic composite image.

The application is composed of three principal layers: a browser-based user interface, a FastAPI HTTP server, and a Celery-based background worker that executes GPU inference. Results are cached in Redis to eliminate redundant computation on repeated requests.

---

## 2. Core Model: CatVTON

### 2.1 Architecture

The inference engine is built on CatVTON (Concatenation-based Virtual Try-On Network), a diffusion model architecture designed specifically for clothing transfer tasks. CatVTON operates by concatenating the person image and the garment image along the channel dimension and passing the combined representation through a denoising U-Net conditioned on the clothing region mask.

The base checkpoint used is a Stable Diffusion inpainting model (`booksforcharlie/stable-diffusion-inpainting`). CatVTON fine-tunes the cross-attention layers to attend jointly to both the person context and the garment condition, enabling semantically coherent texture transfer without explicit warping.

### 2.2 Masking Strategy

Accurate segmentation of the clothing region on the person image is critical to the quality of the output. The system uses two models in combination to produce the inpainting mask:

- **DensePose (Detectron2):** Estimates the body surface correspondence map, providing dense UV coordinates that identify which body parts are visible and where on the image they are located.
- **SCHP (Self-Correcting Human Parsing):** A human parsing model that produces pixel-level semantic segmentation labels for clothing categories including upper body, lower body, and full-body regions.

The `AutoMasker` class combines these two signals to produce a binary mask that precisely delineates the region to be replaced. The `mask_type` parameter controls which region is targeted: `upper`, `lower`, or `overall`.

### 2.3 Inference Parameters

| Parameter | Description |
|---|---|
| `num_inference_steps` | Number of denoising steps. Range: 5 to 50. Higher values produce finer detail at the cost of compute time. |
| `guidance_scale` | Classifier-free guidance scale. Fixed at 2.5 for single-pass inference. |
| `generator` | PyTorch random generator seeded for reproducibility. Multiple seeds produce variation variants. |
| `height / width` | Output resolution fixed at 1024 x 768 pixels. |

---

## 3. Processing Pipeline

### 3.1 Single Garment (upper or lower)

When a single garment category is selected, the pipeline executes a single inpainting pass. The garment image undergoes background removal via `rembg` (u2net model) and is scaled to the target canvas dimensions with padding to preserve aspect ratio. A size scaling factor is applied to simulate garment fit relative to body size before padding.

The inpainting result is composited back onto the original person photograph using a soft-edged Gaussian-blurred version of the mask, preserving the unmasked regions of the person image at full quality.

### 3.2 Full Outfit (overall)

When the `overall` category is selected, the system supports two modes of operation:

- **Single garment provided:** The garment is treated as a full-body item (dress, jumpsuit, etc.) and a single inference pass is performed using the overall body mask, which covers both the upper and lower torso regions simultaneously.
- **Two garments provided (upper and lower):** Two sequential inference passes are performed. The first pass applies the upper garment to the upper body region. The result is then used as the input person image for the second pass, which applies the lower garment to the lower body region. Each pass uses an independently seeded random generator.

### 3.3 Post-Processing: Repaint

After each inference pass, a repaint step composites the diffusion output with the original person image. A Gaussian blur with kernel size proportional to image height (`h/50`, odd) is applied to the mask before blending. This feathers the boundary between the generated garment region and the original photograph, eliminating hard seams at the mask edges.

### 3.4 Optional Upscaling

When upscaling is enabled, the composited output is passed through Real-ESRGAN (`RealESRGAN_x2plus` model) to produce a 2x resolution enhancement. The upscaler operates tile-by-tile (tile size 400, pad 10) to fit within GPU memory constraints. The final output is delivered at approximately 2048 x 1536 pixels.

---

## 4. Garment Category Detection

### 4.1 Motivation

Selecting the correct category is essential for mask accuracy. Uploading a full-length dress with category set to `upper` results in only the torso being replaced, leaving the original lower-body clothing visible. To address this, the system automatically classifies uploaded garments before the user initiates inference.

### 4.2 Implementation

Immediately after a garment image is uploaded, the frontend sends it to the `/detect-category` endpoint. The server forwards the image to Claude Haiku (`anthropic.claude-3-haiku-20240307-v1:0`) on AWS Bedrock using the `boto3` SDK, which leverages the EC2 instance IAM role for authentication without requiring an API key.

The model receives the garment image and a structured classification prompt requesting a single-word response: `upper`, `lower`, or `overall`. The response is validated against these three values before being returned to the frontend. On any failure, the system defaults to `upper` without surfacing an error to the user.

### 4.3 Classification Taxonomy

| Category | Garment Types |
|---|---|
| `upper` | Shirts, t-shirts, blouses, jackets, blazers, hoodies, sweaters, coats, cardigans |
| `lower` | Pants, jeans, trousers, shorts, skirts, leggings, chinos |
| `overall` | Dresses (mini, midi, maxi), jumpsuits, rompers, overalls, full-body suits |

Classification is performed on the raw uploaded image prior to background removal. The model is robust to lifestyle photography, flat-lay product images, and ghost mannequin shots.

---

## 5. System Architecture

### 5.1 Component Overview

| Component | Responsibility |
|---|---|
| `index.html` | Single-page browser interface. Handles image upload, auto-detection, parameter controls, SSE streaming, result display, and variant thumbnails. |
| `server.py` (FastAPI) | Exposes HTTP API endpoints. Validates inputs, calls Bedrock for classification, dispatches Celery tasks, serves SSE streams for job progress. |
| `worker.py` (Celery) | Executes GPU inference. Loads CatVTON pipeline on first task, manages Redis result cache, performs preprocessing, inference, repaint, and upscaling. |
| Redis | Dual-purpose: Celery broker and result backend for task queuing; persistent result cache keyed by SHA-256 hash of input parameters. |
| AWS Bedrock (Claude Haiku) | Garment category classification via vision inference. Invoked per garment upload via `boto3` using the EC2 IAM role. |

### 5.2 Task Execution Model

Celery is configured with a solo pool (`--pool=solo --concurrency=1`) to avoid CUDA re-initialization errors caused by the default prefork pool. In the prefork model, worker child processes are created by forking the parent, and CUDA cannot be re-initialized in a forked subprocess. The solo pool runs tasks in the main process, eliminating this constraint.

The CatVTON pipeline and AutoMasker instances are loaded once on first task execution and held in a module-level singleton. Subsequent tasks reuse the loaded model, avoiding repeated weight loading from disk.

### 5.3 Result Caching

Before inference, the worker computes a SHA-256 hash over the first 64 bytes of the person image base64, garment image base64, category, size, number of steps, and lower garment base64 (if present). This hash is used as a Redis key. If a matching key exists, the cached result is returned immediately without invoking the pipeline. Results are stored with a 1-hour expiration.

### 5.4 API Endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Serves the static frontend (`index.html`). |
| `GET /health` | Health check. Returns `{"status": "ok"}`. |
| `POST /detect-category` | Accepts a garment image, returns the detected category string. |
| `POST /tryon` | Accepts person, garment, and configuration parameters. Returns a list of job IDs. |
| `GET /result/{job_id}` | Polls a single job for its current state and result. |
| `GET /stream/{job_id}` | Server-Sent Events stream that pushes job state updates until completion or failure. |

---

## 6. Size Scaling

The clothing size parameter (XS through XXL) modulates the apparent fit of the garment on the person by scaling the garment image before padding. A scale factor less than 1.0 makes the garment appear tighter; a factor greater than 1.0 makes it appear looser.

| Size | Scale Factor |
|---|---|
| XS | 0.82 |
| S | 0.88 |
| M | 0.94 |
| L | 1.00 |
| XL | 1.06 |
| XXL | 1.12 |

Scaling is applied to the garment image after background removal and before padding to the 768 x 1024 canvas. The scaled garment is centered on a white canvas of the original dimensions, ensuring the padding step sees a consistent input regardless of scale.

---

## 7. Infrastructure and Deployment

### 7.1 Environment

| Property | Value |
|---|---|
| Platform | Amazon EC2, region ap-south-1 (Mumbai) |
| OS | Amazon Linux 2023 |
| Python | 3.11 (Miniconda environment) |
| GPU | CUDA-capable instance required for float16 inference |
| Concurrency | Single worker process, one task at a time |

### 7.2 Dependencies

| Package | Purpose |
|---|---|
| `torch`, `diffusers`, `transformers` | Diffusion inference stack |
| `detectron2` | DensePose body surface estimation |
| `rembg` | Background removal for garment images |
| `basicsr`, `realesrgan` | Real-ESRGAN upscaling |
| `celery`, `redis` | Task queue and result backend |
| `fastapi`, `uvicorn` | HTTP server |
| `boto3` | AWS Bedrock client for garment classification |

### 7.3 Starting the System

```bash
# Start Redis (if not already running as a service)
redis-server --daemonize yes --port 6379

# Terminal 1: HTTP server
python server.py

# Terminal 2: Celery worker
celery -A worker worker --loglevel=info --concurrency=1 --pool=solo

# Clear the result cache when needed
python -c "import redis; redis.from_url('redis://localhost:6379/0').flushdb(); print('cache cleared')"
```

### 7.4 IAM Requirements

The EC2 instance role must have the following Bedrock permissions:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:ListFoundationModels"
  ],
  "Resource": "*"
}
```

The managed policy `AmazonBedrockFullAccess` satisfies these requirements.

---

## 8. Known Limitations

**Lifestyle garment photography.** When the garment image contains a model wearing the item in a complex scene, background removal quality degrades and the garment condition passed to the pipeline may contain residual background elements. Product flat-lay or ghost mannequin images produce significantly better results.

**Pose sensitivity.** Extreme poses, crossed arms, or partial occlusion of the torso reduce mask accuracy and can cause the generated garment to misalign with the body. Frontal, neutral-stance person images yield the best outputs.

**Sequential outfit degradation.** In the two-pass overall mode, the second inference pass operates on the output of the first, which may carry diffusion artifacts. These can compound and reduce lower-body garment quality relative to upper-body quality.

**Single GPU concurrency.** The solo pool processes one request at a time. Concurrent users must queue. Horizontal scaling would require multiple worker instances with separate GPU allocations.

**Footwear and accessories.** The pipeline does not model footwear, bags, hats, or jewellery. These items are preserved unchanged from the original person photograph.

---

*wearit — Internal Technical Documentation*