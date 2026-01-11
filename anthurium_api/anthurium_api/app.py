from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os

app = FastAPI(title="Anthurium Disease Detection API")

# ---- Load model ----
MODEL_PATH = os.path.join(os.path.dirname(__file__), "anthurium_disease_model.keras")
LABELS_PATH = os.path.join(os.path.dirname(__file__), "labels.json")

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model file not found: {MODEL_PATH}")

if not os.path.exists(LABELS_PATH):
    raise RuntimeError(f"labels.json not found: {LABELS_PATH}")

model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)["class_names"]

IMG_SIZE = (224, 224)

# ---- Hardcoded treatments (keys MUST match class_names exactly) ----
TREATMENTS = {
    "healthy": {
        "summary": "Plant looks healthy.",
        "actions": [
            "Keep a consistent watering schedule; avoid overwatering.",
            "Provide bright indirect light and good airflow.",
            "Inspect leaves weekly for early symptoms."
        ],
        "prevention": [
            "Use clean tools and pots.",
            "Avoid keeping leaves wet for long periods."
        ]
    },
    "leaf_spot": {
        "summary": "Leaf spot often worsens with wet leaves and poor airflow.",
        "actions": [
            "Remove infected leaves (discard, don’t compost).",
            "Avoid overhead watering; water at soil level only.",
            "Improve airflow (space plants, use a fan).",
            "If it spreads: use a labeled fungicide/bactericide and follow the product label."
        ],
        "prevention": [
            "Keep foliage dry and reduce humidity spikes.",
            "Disinfect pruning tools after use."
        ]
    },
    "bacterial_blight": {
        "summary": "Bacterial blight spreads quickly in warm, wet conditions.",
        "actions": [
            "Isolate the plant immediately.",
            "Remove severely infected leaves; disinfect tools after every cut.",
            "Avoid splashing water on leaves; reduce humidity if possible.",
            "If needed: consider a copper-based bactericide (only if label-approved for your plant)."
        ],
        "prevention": [
            "Quarantine new plants for 1–2 weeks.",
            "Avoid overcrowding; keep good airflow."
        ]
    },
    "anthracnose": {
        "summary": "Anthracnose is commonly fungal and worsens with wet foliage.",
        "actions": [
            "Remove infected leaves and clean fallen debris.",
            "Keep leaves dry; improve airflow.",
            "If symptoms continue: apply a labeled fungicide (follow label directions)."
        ],
        "prevention": [
            "Water early in the day; avoid wet leaves overnight.",
            "Sanitize pots and tools."
        ]
    },
    "fungal": {
        "summary": "General fungal infection is often linked to high humidity or overwatering.",
        "actions": [
            "Remove infected leaves; dispose safely.",
            "Reduce watering; ensure pot has good drainage.",
            "Improve airflow and keep in bright indirect light.",
            "If spreading: use a labeled fungicide and follow the label."
        ],
        "prevention": [
            "Avoid soggy soil; let top soil dry slightly between waterings.",
            "Keep plant area clean and dry."
        ]
    },
    "root_rot": {
        "summary": "Root rot is usually caused by overwatering and poor drainage.",
        "actions": [
            "Stop watering until the soil partially dries.",
            "Remove plant from pot and inspect roots.",
            "Trim black/mushy roots with sterilized scissors.",
            "Repot into fresh, well-draining mix; clean or replace the pot.",
            "After repotting: water lightly and only when the top soil is partly dry."
        ],
        "prevention": [
            "Use pots with drainage holes.",
            "Never leave the pot sitting in water.",
            "Use a chunky, well-draining soil mix."
        ]
    }
}

@app.get("/")
def root():
    return {
        "status": "ok",
        "message": "Anthurium API is running",
        "classes": class_names
    }

def preprocess_image(image_bytes: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    img = img.resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)
    return arr

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/jpg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Upload a valid image (jpg/png/webp).")

    image_bytes = await file.read()
    x = preprocess_image(image_bytes)

    preds = model.predict(x)[0]  # (num_classes,)

    idx = int(np.argmax(preds))
    disease = class_names[idx]
    confidence = float(np.max(preds)) * 100

    probs = {class_names[i]: float(preds[i]) for i in range(len(class_names))}
    uncertain = confidence < 60.0

    
    treatment = TREATMENTS.get(disease, {
        "summary": "No treatment info available for this disease class.",
        "actions": [],
        "prevention": []
    })

    return JSONResponse({
        "predicted_disease": disease,
        "confidence_percent": round(confidence, 2),
        "uncertain": uncertain,
        "probabilities": probs,
        "treatment": treatment
    })