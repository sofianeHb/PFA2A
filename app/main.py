# main.py

import logging
import time
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
os.makedirs("logs", exist_ok=True)

# Configure le logger pour écrire dans un fichier en mode append
logger = logging.getLogger("predictions_logger")
logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/predictions.log")
handler.setFormatter(logging.Formatter('%(message)s'))
logger.addHandler(handler)
with open("logs/predictions.log", "a") as f:
    f.write("")  # crée le fichier s'il n'existe pas


@app.post("/predict/")
async def predict_pneumonia(file: UploadFile = File(...)):
    start_time = time.time()
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        prediction = predict(img)
        duration = time.time() - start_time

        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "filename": file.filename,
            "prediction": prediction,
            "duration": duration
        }
        print("Prediction logged:", log_entry)  # pour être sûr que le code passe ici
        logger.info(json.dumps(log_entry))

        return JSONResponse(content=prediction)
    except Exception as e:
        print("Logging error:", str(e))

        logger.error(json.dumps({"timestamp": datetime.utcnow().isoformat(), "error": str(e)}))
        return JSONResponse(status_code=500, content={"error": str(e)})
