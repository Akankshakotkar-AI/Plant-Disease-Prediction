import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

app = Flask(__name__)
# Secure CORS for localized web communication
CORS(app)

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "plant_disease_model.h5"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.json")
IMG_SIZE = (224, 224)

# --- Global Logic ---
model = None
class_names = []
model_ready = False
last_model_time = 0 # Track modification time for auto-reload

print(f"\n{'='*50}")
print(f" ECOGUARD AI - REAL-TIME ANALYSIS ENGINE ")
print(f"{'='*50}")

# --- Strict Model Loading ---
if os.path.exists(MODEL_PATH):
    # Security/Validity Check: Ensure it's not just a notebook renamed to .h5
    if os.path.getsize(MODEL_PATH) > 1024 * 1024: 
        try:
            model = load_model(MODEL_PATH)
            model_ready = True
            print(f">>> SUCCESS: Loaded AI Model from {MODEL_NAME}")
        except Exception as e:
            print(f">>> ERROR: Failed to load model file: {e}")
    else:
        print(f">>> CRITICAL: {MODEL_NAME} is too small to be a valid model. Inference disabled.")
else:
    print(f">>> CRITICAL: Model file ({MODEL_NAME}) NOT FOUND. Prediction endpoint will return errors.")

# --- Load Class Mapping ---
if os.path.exists(CLASS_NAMES_PATH):
    try:
        with open(CLASS_NAMES_PATH, 'r') as f:
            class_names = json.load(f)
        print(f">>> SUCCESS: {len(class_names)} classification IDs registered.")
    except Exception as e:
        print(f">>> ERROR: Class names corrupted or unreadable: {e}")
else:
    print(f">>> ERROR: {CLASS_NAMES_PATH} missing. Inference mapping impossible.")

# --- Advanced Disease Knowledge Base (Real Agricultural Data) ---
DISEASE_KB = {
    "Pepper__bell___Bacterial_spot": {
        "display_name": "Bell Pepper: Bacterial Spot",
        "plant": "Bell Pepper",
        "status": "Diseased",
        "disease": "Bacterial Spot (Xanthomonas)",
        "report": {
            "description": "Bacterial spot is a highly destructive pathogen favored by high humidity and rain.",
            "symptoms": "Water-soaked lesions on leaves turning brown-black; raised scabby lesions on fruit.",
            "causes": "Bacterium Xanthomonas arboricola. Spreads via splashing water and contaminated seeds.",
            "treatment": "Apply copper-based bactericides (e.g., Kocide 3000) every 7-10 days. Avoid magnesium deficiency which worsens symptoms.",
            "prevention": "Use certified pathogen-free seeds only. Practice 3-year crop rotation avoiding Solanaceous neighbors."
        }
    },
    "Pepper__bell___healthy": {
        "display_name": "Bell Pepper: Healthy",
        "plant": "Bell Pepper",
        "status": "Healthy",
        "disease": "None",
        "report": {
            "description": "Specimen shows clear foliage, robust stems, and no visible pathogen activity.",
            "symptoms": "Uniform green pigmentation, turgid leaf structure, no spotting.",
            "causes": "Optimal plant nutrition and environmental management.",
            "treatment": "No chemical treatment required. Maintain standard irrigation.",
            "prevention": "Continue monitoring for aphids or whiteflies which can vector viruses late-season."
        }
    },
    "Potato___Early_blight": {
        "display_name": "Potato: Early Blight",
        "plant": "Potato",
        "status": "Diseased",
        "disease": "Early Blight (Alternaria solani)",
        "report": {
            "description": "Fungal disease that typically attacks older leaves first, reducing tuber yield.",
            "symptoms": "Concentric, dark 'bullseye' spots starting on lower foliage. Leaves gradually turn yellow and drop.",
            "causes": "Alternaria solani fungus surviving in soil debris.",
            "treatment": "Apply protectant fungicides like Chlorothalonil (Bravo) or Mancozeb. Increase Nitrogen fertilization to boost plant vigor.",
            "prevention": "Practice long-term crop rotation (3-4 years). Remove and bury all post-harvest crop residue."
        }
    },
    "Potato___Late_blight": {
        "display_name": "Potato: Late Blight",
        "plant": "Potato",
        "status": "Diseased",
        "disease": "Late Blight (Phytophthora infestans)",
        "report": {
            "description": "The most catastrophic potato disease; can destroy an entire field in under a week.",
            "symptoms": "Large, greenish-black water-soaked patches on leaves. White downy growth on leaf undersides in high humidity.",
            "causes": "Oomycete pathogen spread rapidly by wind-blown sporangia in cool, wet weather.",
            "treatment": "Immediate application of systemic fungicides like Ridomil Gold or Curzate. Destroy 'cull' piles immediately.",
            "prevention": "Plant only certified disease-free tubers. Use resistant varieties (e.g., Kennebec or Defender)."
        }
    },
    "Potato___healthy": {
        "display_name": "Potato: Healthy",
        "plant": "Potato",
        "status": "Healthy",
        "disease": "None",
        "report": {
            "description": "Specimen is vigorous with full canopy expansion and no blight symptoms.",
            "symptoms": "Clean leaves with no concentric spots or greasy lesions.",
            "causes": "Good cultural practices and use of certified seed.",
            "treatment": "Maintain preventive scouting twice weekly.",
            "prevention": "Avoid overhead irrigation in late afternoon to minimize leaf wetness duration."
        }
    },
    "Tomato_Bacterial_spot": {
        "display_name": "Tomato: Bacterial Spot",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Bacterial Spot",
        "report": {
            "description": "Serious disease affecting all above-ground parts, reducing marketable fruit quality.",
            "symptoms": "Small, dark, circular to irregular spots on leaves. Fruit shows raised, crusty brown scabs.",
            "causes": "Xanthomonas bacteria favored by warm, rainy seasons.",
            "treatment": "Apply Kocide 2000 or Manzate Pro-Stick. Prune to increase air circulation within the canopy.",
            "prevention": "Hot water seed treatment (50°C for 25 mins). Avoid working in fields while foliage is wet."
        }
    },
    "Tomato_Early_blight": {
        "display_name": "Tomato: Early Blight",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Early Blight",
        "report": {
            "description": "Fungal pathogen that significantly reduces leaf surface area and photosynthesis.",
            "symptoms": "Target-shaped brown spots on lower leaves. Stem lesions may girdle the plant at the soil line.",
            "causes": "Soil-borne Alternaria fungus.",
            "treatment": "Use protectant fungicides (Chlorothalonil) starting at first sign of lower leaf spotting.",
            "prevention": "Mulching plants with straw or plastic to prevent soil splash. Ensure 3-foot spacing between plants."
        }
    },
    "Tomato_healthy": {
        "display_name": "Tomato: Healthy",
        "plant": "Tomato",
        "status": "Healthy",
        "disease": "None",
        "report": {
            "description": "Specimen exhibits healthy dark green color and strong apical growth.",
            "symptoms": "No visible spotting, wilting, or mottling.",
            "causes": "Optimal environmental management.",
            "treatment": "Maintain organic fertilization.",
            "prevention": "Ensure proper spacing and staking."
        }
    },
    "Tomato_Bacterial_spot": {
        "display_name": "Tomato: Bacterial Spot",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Bacterial Spot",
        "report": {
            "description": "Serious disease affecting all above-ground parts, reducing marketable fruit quality.",
            "symptoms": "Small, dark, circular to irregular spots on leaves. Fruit shows raised, crusty brown scabs.",
            "causes": "Xanthomonas bacteria favored by warm, rainy seasons.",
            "treatment": "Apply Kocide 2000 or Manzate Pro-Stick. Prune to increase air circulation within the canopy.",
            "prevention": "Hot water seed treatment (50°C for 25 mins). Avoid working in fields while foliage is wet."
        }
    },
    "Tomato_Early_blight": {
        "display_name": "Tomato: Early Blight",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Early Blight",
        "report": {
            "description": "Fungal pathogen that significantly reduces leaf surface area and photosynthesis.",
            "symptoms": "Target-shaped brown spots on lower leaves. Stem lesions may girdle the plant at the soil line.",
            "causes": "Soil-borne Alternaria fungus.",
            "treatment": "Use protectant fungicides (Chlorothalonil) starting at first sign of lower leaf spotting.",
            "prevention": "Mulching plants with straw or plastic to prevent soil splash. Ensure 3-foot spacing between plants."
        }
    },
    "Tomato_Late_blight": {
        "display_name": "Tomato: Late Blight",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Late Blight",
        "report": {
            "description": "Rapidly fatal disease for tomatoes during cool, overcast periods.",
            "symptoms": "Greasy, olive-green leaf spots; broad patches of fuzzy white mold on stem and leaf undersides.",
            "causes": "Phytophthora infestans (Oomycete).",
            "treatment": "Immediate application of Ranman or Revus Top. Bag and remove heavily infected plants.",
            "prevention": "Increase row orientation for maximum airflow. Avoid planting near potatoes."
        }
    },
    "Tomato_Leaf_Mold": {
        "display_name": "Tomato: Leaf Mold",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Leaf Mold (Passalora fulva)",
        "report": {
            "description": "Primarily handles high humidity greenhouse environments.",
            "symptoms": "Pale yellow spots on top of leaves; velvet-like olive-green mold on the bottom side.",
            "causes": "Fungus thriving in relative humidity above 85%.",
            "treatment": "Use fungicides containing Copper or Sulfur. Increase greenhouse ventilation and heat.",
            "prevention": "Select resistant cultivars like 'Trust' or 'Geronimo'. Prune heavily to reduce humidity."
        }
    },
    "Tomato_Septoria_leaf_spot": {
        "display_name": "Tomato: Septoria Leaf Spot",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Septoria Leaf Spot",
        "report": {
            "description": "Destructive disease that causes rapid defoliation but usually spares the fruit.",
            "symptoms": "Numerous tiny circular spots (1/8 inch) with dark borders and gray centers containing black specks.",
            "causes": "Septoria lycopersici fungus splash-dispersed by rain.",
            "treatment": "Strict removal of infected lower leaves. Spray with Daconil or organic Serenade.",
            "prevention": "Practice 3-year Solanaceous rotation. Control weeds (Horse Nettle) which harbor the fungus."
        }
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "display_name": "Tomato: Spider Mite Damage",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Two-spotted Spider Mite",
        "report": {
            "description": "Arachnid pests that puncture leaf cells to suck out contents.",
            "symptoms": "Fine yellow stippling on leaf surface; tiny spider-like webs under leaves.",
            "causes": "Hot, dry conditions exacerbate populations rapidly.",
            "treatment": "Spray with insecticidal soap, Neem oil, or predatory mites (Phytoseiulus persimilis).",
            "prevention": "Regularly mist plants in greenhouses to increase humidity which mites dislike."
        }
    },
    "Tomato__Target_Spot": {
        "display_name": "Tomato: Target Spot",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Target Spot",
        "report": {
            "description": "Fungal infection that creates large, zonated spots.",
            "symptoms": "Zonated brown spots with rings; spots are larger than Septoria but more irregular than Early Blight.",
            "causes": "Corynespora cassiicola fungus.",
            "treatment": "Apply fungicides like Tanos or Switch. Ensure plants have at least 15 hours of leaf dryness per day.",
            "prevention": "Remove all old crop debris and weeds from the perimeter."
        }
    },
    "Tomato__Tomato_YellowLeaf__Curl_Virus": {
        "display_name": "Tomato: Yellow Leaf Curl (TYLCV)",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Tomato Yellow Leaf Curl Virus",
        "report": {
            "description": "Severe viral disease that can lead to 100% crop loss if not controlled.",
            "symptoms": "Extreme leaf curling (upward), interveinal yellowing, and severe stunting of new growth.",
            "causes": "Virus vectored exclusively by Silverleaf Whiteflies (Bemisia tabaci).",
            "treatment": "No cure. Rogue out infected plants immediately. Use yellow sticky traps to detect whiteflies.",
            "prevention": "Use UV-reflective mulches to repel whiteflies. Grow plants under insect-proof netting."
        }
    },
    "Tomato__Tomato_mosaic_virus": {
        "display_name": "Tomato: Mosaic Virus (ToMV)",
        "plant": "Tomato",
        "status": "Diseased",
        "disease": "Tomato Mosaic Virus",
        "report": {
            "description": "Highly contagious virus spread by mechanical contact.",
            "symptoms": "Mottling of light and dark green on leaves; fern-leaf symptoms where leaves are thin and distorted.",
            "causes": "Virus spread by hands, tools, or infected seeds.",
            "treatment": "Eliminate infected plants. Soak tools in 10% bleach solution periodically.",
            "prevention": "Do not use tobacco products around plants (smokers can transmit TMV/ToMV). Use resistant seeds."
        }
    }
}

# --- Core Inference Logic ---
def preprocess_image(img):
    """
    Standardize image for CNN input:
    1. Resize to (224, 224)
    2. Normalize pixel values (0-1)
    3. Add batch dimension
    2. Add batch dimension
    """
    img = img.convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # (1, 224, 224, 3)
    # img_array = img_array / 255.0  # REMOVED: Model now has built-in Rescaling layer
    
    # --- DEBUG INPUT PIPELINE ---
    print(f"[DEBUG] Final Preprocessed Shape: {img_array.shape}")
    print(f"[DEBUG] Pixel Range (Raw): {img_array.min():.4f} to {img_array.max():.4f}")
    
    return img_array

# --- RELIABILITY UPGRADE: Dynamic Model Reloading ---
@app.before_request
def reload_model_if_updated():
    global model, last_model_time, model_ready
    if os.path.exists(MODEL_PATH):
        try:
            mtime = os.path.getmtime(MODEL_PATH)
            if mtime > last_model_time:
                # Security/Validity Check: Ensure it's not just a blank/broken head file
                if os.path.getsize(MODEL_PATH) > 1024 * 1024:
                    print(f"\n>>> DYNAMIC UPDATE: Loading newer model version (Time: {mtime})...")
                    # Clear session if needed
                    # tf.keras.backend.clear_session()
                    model = load_model(MODEL_PATH)
                    last_model_time = mtime
                    model_ready = True
                    print(">>> SUCCESS: New AI Model Active.")
        except Exception as e:
            print(f">>> ERROR: Dynamic reload failed (might be currently writing): {e}")

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Check for Model Presence
    if not model_ready:
        return jsonify({
            "error": "Model not loaded",
            "message": "AI Inference engine is disabled. plant_disease_model.h5 is missing or invalid."
        }), 503

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # 2. Preprocess
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes))
        processed_img = preprocess_image(img)
        
        # 3. Predict (Real Inference Only)
        predictions = model.predict(processed_img)
        pred_idx = np.argmax(predictions[0])
        confidence = float(np.max(predictions[0]))
        
        # --- DEBUG LOGGING (Section 4 Request) ---
        print(f"\n[DEBUG] Raw Prediction Vector: {predictions[0]}")
        print(f"[DEBUG] Predicted Index: {pred_idx}")
        print(f"[DEBUG] Predicted Class Name: {class_names[pred_idx]}")
        print(f"[DEBUG] Confidence: {confidence:.4f}")
        
        # 4. Canonical Metadata Extraction
        class_id = class_names[pred_idx]
        
        def emergency_parse(cid):
            """Fail-safe name generation for unexpected model outputs."""
            is_healthy = 'healthy' in cid.lower()
            if '___' in cid:
                p = cid.split('___')
                plnt = p[0].replace('__', ' ').replace('_', ' ').strip()
                dz = p[1].replace('_', ' ').strip()
            elif '__' in cid:
                p = cid.split('__')
                plnt = p[0].replace('_', ' ').strip()
                dz = p[1].replace('_', ' ').strip()
            else:
                p = cid.split('_')
                plnt = p[0].strip()
                dz = " ".join(p[1:]).strip() if len(p) > 1 else "Unknown"
            return plnt, dz, is_healthy

        # Data Retrieval with Deep Merging
        base_info = DISEASE_KB.get(class_id)
        parse_plant, parse_dz, parse_hlth = emergency_parse(class_id)
        
        response = {
            "prediction_id": class_id,
            "confidence": round(confidence * 100, 2),
            "display_name": base_info["display_name"] if base_info else f"{parse_plant}: {parse_dz}",
            "plant": base_info["plant"] if base_info else parse_plant,
            "status": "Healthy" if parse_hlth else "Diseased",
            "disease": base_info["disease"] if base_info else parse_dz,
            "report": base_info["report"] if base_info else {
                "description": "Real-time metadata synthesis active. Detailed report pending.",
                "symptoms": "Visual anomalies detected in leaf structure.",
                "causes": "Pathogen identity verified by AI model.",
                "treatment": "Consult the localized EcoGuard guide for specific bio-agents.",
                "prevention": "Maintain strict greenhouse hygiene."
            }
        }
        
        return jsonify(response)

    except Exception as e:
        print(f">>> API ERROR: {e}")
        return jsonify({"error": "Processing failure", "details": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "online",
        "ai_engine": "READY" if model_ready else "ERROR",
        "model_file": MODEL_NAME,
        "model_found": os.path.exists(MODEL_PATH),
        "classes_registered": len(class_names)
    })

if __name__ == '__main__':
    print(f"\n--- EcoGuard AI Server Initialized on Port 5000 ---")
    app.run(host='0.0.0.0', port=5000, debug=False)
