import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import zipfile
import io
import base64
from PIL import Image
from database import SessionLocal, PredictionHistory
import json
import google.generativeai as genai
from dotenv import load_dotenv
import os
from config import CLASS_NAMES
from model_loader import get_all_models

app = FastAPI()
load_dotenv(".env.local")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    if "X-Content-Type-Options" not in response.headers:
        response.headers["X-Content-Type-Options"] = "nosniff"
    if "X-XSS-Protection" not in response.headers:
        response.headers["X-XSS-Protection"] = "1; mode=block"
    return response

print("Loading models...")
models = get_all_models()
print(f"Loaded {len(models)} models: {list(models.keys())}")

def generate_advice(label_name: str, confidence: float, probs: list) -> str:
    prompt = (
        "You are an agricultural assistant specialized in diagnosing nutrient deficiencies "
        "in rice leaves, including Nitrogen (N), Phosphorus (P), and Potassium (K) deficiency. "
        f"Detected condition: {label_name}. "
        f"Model confidence: {confidence:.2f}. "
        f"Class probabilities: {probs}. "
        "Provide a brief, specific, actionable recommendation for rice farmers. "
        "Include only practical steps such as fertilizer adjustment, monitoring, and prevention. "
        "Do NOT include greeting, disclaimers, or formatting. "
        "Return plain text only (2–3 concise sentences)."
    )
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Could not retrieve advice at this time. Error: {str(e)}"
    
def save_prediction_to_db(result: dict):
    db = SessionLocal()
    try:
        record = PredictionHistory(
            file_name=result.get("fileName", "unknown"),
            label=result["label"],
            label_name=result["label_name"],
            confidence=result["confidence"],
            probs=json.dumps(result.get("probs", [])),
            advice=result.get("advice", ""),
            image_data=result.get("image_data", "")
        )
        db.add(record)
        db.commit()
    finally:
        db.close()

def is_image_file(filename: str) -> bool:
    """Kiểm tra đuôi file có phải là ảnh không"""
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

@app.get("/models")
async def get_models():
    models_list = []
    if 'xception' in models:
        models_list.append({"key": "xception", "name": "Xception"})
    if 'resnet50' in models:
        models_list.append({"key": "resnet50", "name": "ResNet50"})
    if 'efficientnet' in models:
        models_list.append({"key": "efficientnet", "name": "EfficientNetB0"})
    if 'mobilenet' in models:
        models_list.append({"key": "mobilenet", "name": "MobileNetV3"})
    return models_list

@app.post("/predict-image/{model_key}")
async def predict_image(model_key: str, file: UploadFile = File(...)):
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    selected_model = models[model_key]
    try:
        image_bytes = await file.read()
        # Convert image to base64 for storage
        image_base64 = base64.b64encode(image_bytes).decode('utf-8')
        
        preds = selected_model.predict(image_bytes)
        
        probs = preds[0]
        label_idx = int(np.argmax(probs))
        confidence = float(np.max(probs))
        label_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else "Unknown"
        advice = generate_advice(label_name, confidence, probs.tolist())

        result = {
            "label": label_idx,
            "fileName": file.filename,
            "label_name": label_name,
            "confidence": confidence,
            "probs": probs.tolist(),
            "advice": advice,
            "image_data": image_base64
        }
        save_prediction_to_db(result)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch-zip/{model_key}")
async def predict_batch_zip(model_key: str, file: UploadFile = File(...)):
    """Xử lý file ZIP chứa nhiều ảnh"""
    
    if model_key not in models:
        raise HTTPException(status_code=404, detail=f"Model '{model_key}' not found")
    
    selected_model = models[model_key]
    
    if not file.filename.endswith(".zip"):
        raise HTTPException(status_code=400, detail="File must be a ZIP archive")

    results = []
    
    try:
        zip_bytes = await file.read()
        
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zip_ref:
            file_list = zip_ref.namelist()
            
            image_files = [f for f in file_list if is_image_file(f) and not f.startswith("__MACOSX")]

            if not image_files:
                raise HTTPException(status_code=400, detail="No valid images found in ZIP file")

            images_data = []
            for filename in image_files:
                try:
                    with zip_ref.open(filename) as img_file:
                        img_bytes = img_file.read()
                        images_data.append({
                            "filename": filename,
                            "bytes": img_bytes,
                            "base64": base64.b64encode(img_bytes).decode('utf-8')
                        })
                except Exception as e:
                    raise HTTPException(status_code=500, detail=f"Error reading file {filename}: {str(e)}")
            
            batch_preds = []
            for img_data in images_data:
                pred = selected_model.predict(img_data["bytes"])
                batch_preds.append(pred[0])
            
            batch_preds = np.array(batch_preds)
            
            for i, img_data in enumerate(images_data):
                probs = batch_preds[i]
                label_idx = int(np.argmax(probs))
                confidence = float(np.max(probs))
                label_name = CLASS_NAMES[label_idx] if label_idx < len(CLASS_NAMES) else "Unknown"
                advice = generate_advice(label_name, confidence, probs.tolist())
                
                result = {
                    "label": label_idx,
                    "fileName": img_data["filename"],
                    "label_name": label_name,
                    "confidence": confidence,
                    "probs": probs.tolist(),
                    "advice": advice,
                    "image_data": img_data["base64"]
                }
                save_prediction_to_db(result)
                results.append(result)
                            
        return results

    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

@app.get("/history")
async def get_history():
    db = SessionLocal()
    try:
        records = db.query(PredictionHistory).order_by(PredictionHistory.created_at.desc()).all()
        return [
            {
                "fileName": r.file_name,
                "label": r.label,
                "label_name": r.label_name,
                "confidence": r.confidence,
                "probs": json.loads(r.probs),
                "advice": r.advice,
                "image_data": r.image_data,
                "created_at": r.created_at.isoformat()
            } for r in records
        ]
    finally:
        db.close()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)