import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import random

# --- Configuration ---
DATA_DIR = r"c:/Users/garim/OneDrive/Desktop/ML_plant/PlantVillage"
MODEL_PATH = "plant_disease_model.h5"
CLASS_NAMES_PATH = "class_names.json"
IMG_SIZE = (224, 224)
SAMPLE_SIZE = 50
CONFIDENCE_THRESHOLD = 0.60

def validate():
    print(f"\n--- ACCURACY & RELIABILITY VALIDATION STARTING ---")
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file {MODEL_PATH} not found. Finish training first.")
        return

    # 1. Load Resources
    print("Loading model and class names...")
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_NAMES_PATH, 'r') as f:
        class_names = json.load(f)

    # 2. Collect Random Samples
    print(f"Selecting {SAMPLE_SIZE} random samples from the dataset...")
    all_image_paths = []
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_paths.append(os.path.join(root, file))
    
    samples = random.sample(all_image_paths, min(SAMPLE_SIZE, len(all_image_paths)))

    # 3. Perform Inference
    correct = 0
    confident_and_correct = 0
    low_confidence_count = 0
    
    print(f"{'Actual Class':<40} | {'Predicted':<40} | {'Conf%':<6} | {'Status'}")
    print("-" * 110)

    for img_path in samples:
        # Get actual class from folder name
        actual_class = os.path.basename(os.path.dirname(img_path))
        
        # Preprocess
        img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0
        
        # Predict
        preds = model.predict(img_array, verbose=0)
        pred_idx = np.argmax(preds[0])
        confidence = float(np.max(preds[0]))
        predicted_class = class_names[pred_idx]
        
        # Scoring
        is_correct = (actual_class == predicted_class)
        is_confident = (confidence >= CONFIDENCE_THRESHOLD)
        
        if is_correct: correct += 1
        if is_correct and is_confident: confident_and_correct += 1
        if not is_confident: low_confidence_count += 1
        
        status_str = "OK" if is_correct and is_confident else ("LOW CONF" if not is_confident else "WRONG")
        
        print(f"{actual_class[:40]:<40} | {predicted_class[:40]:<40} | {confidence*100:>5.1f}% | {status_str}")

    # 4. Summary report
    print("-" * 110)
    print(f"Validation Complete on {len(samples)} images.")
    print(f"Overall Accuracy: {(correct/len(samples))*100:.1f}%")
    print(f"Reliable Detections (Correct & >60% Conf): {(confident_and_correct/len(samples))*100:.1f}%")
    print(f"Low Confidence Rejections: {low_confidence_count}")
    print(f"Target Reliability: >85% Recommended for Real-World Deployment.")

if __name__ == "__main__":
    validate()
