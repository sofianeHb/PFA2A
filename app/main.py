from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
import logging
import json
from datetime import datetime
from model.pneumonia_model import predict

app = FastAPI()

# Logger JSON structuré
logger = logging.getLogger("inference_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(message)s')  # ligne brute pour Loki
handler.setFormatter(formatter)
logger.addHandler(handler)

@app.post("/predict/")
async def predict_pneumonia(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        prediction = predict(img)

        # Log structuré : timestamp, nom de fichier, prédiction
        logger.info(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "filename": file.filename,
            "prediction": prediction,
            "model": "Enhanced_model_V2.keras"
        }))

        return JSONResponse(content=prediction)
    except Exception as e:
        logger.error(json.dumps({
            "timestamp": datetime.utcnow().isoformat(),
            "filename": file.filename if file else "unknown",
            "error": str(e)
        }))
        return JSONResponse(status_code=500, content={"error": str(e)})
