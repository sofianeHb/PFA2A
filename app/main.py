from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from PIL import Image
import io
from model.pneumonia_model import predict

app = FastAPI()

# Endpoint pour vérifier que le serveur est actif
@app.get("/")
async def root():
    return {"message": "FastAPI is running "}

# Endpoint pour prédire la pneumonie
@app.post("/predict/")
async def predict_pneumonia(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        prediction = predict(img)
        return JSONResponse(content=prediction)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
