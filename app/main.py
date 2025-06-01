# main.py

import logging
import time
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from model.pneumonia_model import predict
import os
import json

app = FastAPI()

# Crée le dossier logs s'il n'existe pas
LOG_DIR = "/app/logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, "predictions.log")

# Configure le logger pour écrire dans un fichier en mode append
logger = logging.getLogger("predictions_logger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler(log_file_path)
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)

with open(log_file_path, "a") as f:
    f.write("")  # crée le fichier s'il n'existe pas


@app.post("/predict/")
async def predict_pneumonia(file: UploadFile = File(...)):
    request_id = str(uuid.uuid4())
    start_time = time.time()

    logger.info(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "event": "Request received",
        "request_id": request_id,
        "filename": file.filename
    }))

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        logger.info(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "Image format check",
            "request_id": request_id,
            "mode": img.mode,
            "size": img.size
        }))

        logger.info(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "Prediction started",
            "request_id": request_id,
            "filename": file.filename
        }))

        prediction = predict(img)
        duration = time.time() - start_time

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": "Prediction successful",
            "request_id": request_id,
            "filename": file.filename,
            "prediction": prediction,
            "duration": duration,
            "status": 200
        }

        print("Prediction logged:", log_entry)
        logger.info(json.dumps(log_entry))

        return JSONResponse(content=prediction)

    except Exception as e:
        logger.error(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "event": "Prediction failed",
            "request_id": request_id,
            "filename": file.filename if file else "unknown",
            "error": str(e),
            "status": 500
        }))

        return JSONResponse(status_code=500, content={"error": str(e)})
