import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image
import json
import os
from build import build_custom_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import img_to_array


H, W = 224, 224
IMG_SIZE = (H, W)
BATCH_SIZE = 32
CLASS_NAMES = ['NORMAL', 'PNEUMONIA']  # Example class labels pneumonia
CLASS_COUNT = len(CLASS_NAMES)
IMG_SHAPE = (H, W, 3)
LEARNING_RATE = 0.0001    
model= build_custom_model(
        EfficientNetB0,
        IMG_SHAPE,
        CLASS_COUNT,
        freeze_percentage=0,
        weights="imagenet",
        pooling="max",
        learning_rate=LEARNING_RATE,
        plot_file="model_plot.png",
        show_summary=False
)
model.load_weights("./outputs/models/Enhanced_model_V2.keras")

# Charger le mapping des classes
with open("model/class_mapping.json", "r") as f:
    class_mapping = json.load(f)  # {"NORMAL": 0, "PNEUMONIA": 1, ...}
    class_mapping = {v: k for k, v in class_mapping.items()}  # inverse

def predict(img: Image.Image, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # EfficientNet preprocessing

    preds = model.predict(img_array)
    class_idx = np.argmax(preds, axis=1)[0]
    class_label = class_mapping[class_idx]

    return {"class": class_label, "confidence": float(np.max(preds))}
