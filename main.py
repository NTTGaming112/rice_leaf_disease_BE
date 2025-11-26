import numpy as np

from fastapi import FastAPI, File, UploadFile
from model_loader import model
from config import CLASS_NAMES
from inference_utils import preprocess_image_for_model

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    input_array = preprocess_image_for_model(image_bytes)
    preds = model.predict(input_array)
    probs = preds[0]
    label = int(np.argmax(probs))
    confidence = float(np.max(probs))
    label_name = CLASS_NAMES[label]
    return {
        "label": label,
        "label_name": label_name,
        "confidence": confidence,
        "probs": probs.tolist()
    }

