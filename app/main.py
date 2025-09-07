from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

from app.schemas import DetectionResponse, ErrorResponse, Detection, BoundingBox
from app.inference import detect_logos

app = FastAPI(
    title="T-Bank Logo Detector",
    description="REST API сервис для поиска логотипа Т-Банка",
    version="0.1.0"
)


@app.post("/detect", response_model=DetectionResponse, responses={400: {"model": ErrorResponse}})
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении.
    """
    try:
        contents = await file.read()
        detections_raw = detect_logos(contents)

        detections = [
            Detection(bbox=BoundingBox(**d["bbox"]))
            for d in detections_raw
        ]

        return DetectionResponse(detections=detections)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health():
    """Проверка здоровья сервиса"""
    return {"status": "ok"}
