import os
import uuid
import io
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
import joblib
from PIL import Image

from app import config
from app.engine import TeachableEngine

app = FastAPI(title="Teachable Machine - TensorFlow Backend")
engine = TeachableEngine()

@app.post("/upload-sample")
async def upload_sample(class_name: str = Form(...), files: List[UploadFile] = File(...)):
    """Creates a local target folder and saves inbound images securely."""
    clean_name = "".join(c for c in class_name if c.isalnum() or c in ("-", "_")).strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Invalid character names in class field.")
        
    class_folder = os.path.join(config.DATA_DIR, clean_name)
    os.makedirs(class_folder, exist_ok=True)
    
    saved_count = 0
    for file in files:
        try:
            # Read streaming file bytes asynchronously
            contents = await file.read()
            img = Image.open(io.BytesIO(contents))
            
            # Use random tokens for filenames to prevent conflict or image overriding
            unique_filename = f"{uuid.uuid4().hex}.jpg"
            img.convert("RGB").save(os.path.join(class_folder, unique_filename), "JPEG")
            saved_count += 1
        except Exception:
            continue
            
    return {"message": f"Successfully cached {saved_count} frames to class '{clean_name}'"}

@app.post("/train")
def train_model():
    """Triggers on-the-fly local ML processing."""
    try:
        msg = engine.train_classifier()
        return {"status": "success", "message": msg}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server Runtime Error: {str(e)}")

@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """Calculates categorization probabilities for an active testing frame."""
    if not os.path.exists(config.MODEL_PATH):
        raise HTTPException(status_code=400, detail="No trained model found. Run a training cycle first.")
        
    try:
        classifier = joblib.load(config.MODEL_PATH)
        temp_path = os.path.join(config.MODEL_DIR, "temp_inference.jpg")
        
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        img.convert("RGB").save(temp_path, "JPEG")
        
        # Feature extraction and prediction mapping
        features = engine.extract_features(temp_path)
        probabilities = classifier.predict_proba([features])[0]
        classes = classifier.classes_
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
        # Structure the target metrics into an explicit dictionary map
        prob_dict = {classes[i]: float(probabilities[i]) for i in range(len(classes))}
        prediction = classifier.predict([features])[0]
        
        return {"prediction": prediction, "probabilities": prob_dict}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference crash: {str(e)}")