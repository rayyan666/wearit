import io
import json
import asyncio
import base64
import boto3
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse

from worker import tryon_task, app as celery_app

app = FastAPI(title="wearit")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Bedrock client (uses EC2 IAM role — no API key needed) ───────────────────
_bedrock_runtime = None

def get_bedrock_runtime():
    global _bedrock_runtime
    if _bedrock_runtime is None:
        _bedrock_runtime = boto3.client("bedrock-runtime", region_name="ap-south-1")
    return _bedrock_runtime


# Confirmed working in ap-south-1 (from test_bedrock_models.py run)
DETECTION_MODELS = [
    "anthropic.claude-3-haiku-20240307-v1:0",   # fast, cheap, confirmed working
    "anthropic.claude-3-sonnet-20240229-v1:0",  # fallback if haiku fails
]

DETECTION_PROMPT = (
    "Classify this garment image. Reply with exactly one word:\n"
    "- 'upper' for shirts, tops, t-shirts, blouses, jackets, hoodies, sweaters, coats\n"
    "- 'lower' for pants, jeans, shorts, skirts, trousers, leggings\n"
    "- 'overall' for dresses, maxi dresses, midi dresses, mini dresses, "
    "jumpsuits, overalls, rompers, full-body suits\n"
    "Reply with only that single word, nothing else."
)

# Mime types accepted by Bedrock vision
ALLOWED_MIME = {"image/jpeg", "image/png", "image/gif", "image/webp"}


async def detect_garment_category(garment_b64: str, mime_type: str = "image/jpeg") -> str:
    """
    Classify the garment using AWS Bedrock Claude Vision.
    Tries models in DETECTION_MODELS order, returns 'upper' on total failure.
    """
    if mime_type not in ALLOWED_MIME:
        mime_type = "image/jpeg"

    client = get_bedrock_runtime()
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": [{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": mime_type,
                        "data": garment_b64,
                    },
                },
                {"type": "text", "text": DETECTION_PROMPT},
            ],
        }],
    })

    for model_id in DETECTION_MODELS:
        try:
            resp   = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            result = json.loads(resp["body"].read())
            text   = result["content"][0]["text"].strip().lower()
            category = text if text in ("upper", "lower", "overall") else "upper"
            print(f"[detect-category] model={model_id} → '{category}'")
            return category
        except Exception as e:
            print(f"[detect-category] {model_id} failed: {e}")
            continue

    print("[detect-category] all models failed, defaulting to 'upper'")
    return "upper"


# ── routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/detect-category")
async def detect_category_endpoint(garment: UploadFile = File(...)):
    """
    Lightweight endpoint called by the frontend right after garment upload.
    Returns {"category": "upper"|"lower"|"overall"} so the UI can auto-select.
    """
    try:
        data        = await garment.read()
        garment_b64 = base64.b64encode(data).decode()
        mime_type   = garment.content_type or "image/jpeg"
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image file")

    category = await detect_garment_category(garment_b64, mime_type)
    return {"category": category}


@app.post("/tryon")
async def tryon(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    lower_garment: Optional[UploadFile] = File(None),
    category: str = Form("upper"),
    steps: int = Form(20),
    size: str = Form("M"),
    num_results: int = Form(1),
    use_upscale: bool = Form(True),
    auto_category: bool = Form(False),
):
    if category not in ("upper", "lower", "overall", "inner", "outer"):
        raise HTTPException(status_code=400, detail="invalid category")
    if steps < 5 or steps > 50:
        raise HTTPException(status_code=400, detail="steps must be 5-50")
    if size.upper() not in ("XS", "S", "M", "L", "XL", "XXL"):
        raise HTTPException(status_code=400, detail="invalid size")
    if num_results < 1 or num_results > 3:
        raise HTTPException(status_code=400, detail="num_results must be 1-3")

    try:
        person_b64    = base64.b64encode(await person.read()).decode()
        garment_bytes = await garment.read()
        garment_b64   = base64.b64encode(garment_bytes).decode()
        lower_b64 = None
        if lower_garment and lower_garment.filename:
            data = await lower_garment.read()
            if data:
                lower_b64 = base64.b64encode(data).decode()
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image files")

    detected_category = category
    if auto_category:
        mime_type = garment.content_type or "image/jpeg"
        detected_category = await detect_garment_category(garment_b64, mime_type)

    seeds = [42, 123, 777][:num_results]
    tasks = [
        tryon_task.delay(
            person_b64=person_b64,
            garment_b64=garment_b64,
            category=detected_category,
            lower_b64=lower_b64,
            steps=steps,
            size=size.upper(),
            seed=seed,
            use_upscale=use_upscale,
        )
        for seed in seeds
    ]

    return {
        "job_ids": [t.id for t in tasks],
        "detected_category": detected_category,
    }


@app.get("/result/{job_id}")
async def get_result(job_id: str):
    task = celery_app.AsyncResult(job_id)

    if task.state == "PENDING":
        return {"status": "pending", "step": "queued"}
    if task.state == "PROGRESS":
        return {"status": "processing", "step": (task.info or {}).get("step", "")}
    if task.state == "SUCCESS":
        result = task.result
        return {"status": "done", "image": result["image"], "cached": result.get("cached", False)}
    if task.state == "FAILURE":
        return {"status": "error", "detail": str(task.info)}

    return {"status": task.state.lower()}


@app.get("/stream/{job_id}")
async def stream_result(job_id: str):
    async def event_generator():
        task      = celery_app.AsyncResult(job_id)
        last_step = None

        while True:
            state = task.state

            if state == "PENDING":
                data = json.dumps({"status": "pending", "step": "queued"})
                yield f"data: {data}\n\n"

            elif state == "PROGRESS":
                step = (task.info or {}).get("step", "")
                if step != last_step:
                    last_step = step
                    data = json.dumps({"status": "processing", "step": step})
                    yield f"data: {data}\n\n"

            elif state == "SUCCESS":
                result = task.result
                data = json.dumps({
                    "status": "done",
                    "image": result["image"],
                    "cached": result.get("cached", False),
                })
                yield f"data: {data}\n\n"
                break

            elif state == "FAILURE":
                data = json.dumps({"status": "error", "detail": str(task.info)})
                yield f"data: {data}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)