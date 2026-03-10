import io
import base64
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from PIL import Image

from pipeline import get_pipeline

app = FastAPI(title="wearit")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/tryon")
async def tryon(
    person: UploadFile = File(...),
    garment: UploadFile = File(...),
    lower_garment: Optional[UploadFile] = File(None),
    category: str = Form("upper"),
    steps: int = Form(20),
    size: str = Form("M"),
):
    if category not in ("upper", "lower", "overall", "inner", "outer"):
        raise HTTPException(status_code=400, detail="category must be upper, lower, overall, inner, or outer")

    if steps < 5 or steps > 50:
        raise HTTPException(status_code=400, detail="steps must be between 5 and 50")

    if size.upper() not in ("XS", "S", "M", "L", "XL", "XXL"):
        raise HTTPException(status_code=400, detail="size must be one of XS, S, M, L, XL, XXL")

    try:
        person_img = Image.open(io.BytesIO(await person.read())).convert("RGB")
        garment_img = Image.open(io.BytesIO(await garment.read())).convert("RGB")
        lower_img = None
        if lower_garment and lower_garment.filename:
            lower_data = await lower_garment.read()
            if lower_data:
                lower_img = Image.open(io.BytesIO(lower_data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="invalid image files")

    pipe = get_pipeline()

    try:
        result = pipe.run(
            person_image=person_img,
            upper_garment=garment_img,
            category=category,
            lower_garment=lower_img,
            steps=steps,
            size=size,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()

    return {"image": f"data:image/png;base64,{encoded}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8001, reload=False)