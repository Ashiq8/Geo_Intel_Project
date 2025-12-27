import os
import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- SETTINGS ---
MODEL_PATH = "geo_intel_model.h5"
DATA_DIR = "processed_data/train_images/"  # Taking from our processed tiles
MASK_DIR = "processed_data/train_masks/"
SAVE_PATH = "prediction_results.png"
NUM_SAMPLES = 5  # How many images to test

def predict_and_plot():
    # 1. Load the Trained Brain
    print(f"⏳ Loading model from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model file not found! Did training finish?")
        return
        
    model = load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")

    # 2. Pick Random Images
    all_images = os.listdir(DATA_DIR)
    sample_images = random.sample(all_images, NUM_SAMPLES)

    # 3. Setup the Plot (Report Card)
    fig, axes = plt.subplots(NUM_SAMPLES, 3, figsize=(12, NUM_SAMPLES * 4))
    # Columns: Input Image | Real Answer (Ground Truth) | AI Prediction

    print("🚀 Starting predictions...")
    
    for i, img_name in enumerate(sample_images):
        # -- Prepare Input --
        img_path = os.path.join(DATA_DIR, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For display
        
        # Normalize for AI (0-1 range)
        img_input = img / 255.0
        img_input = np.expand_dims(img_input, axis=0) # Shape: (1, 256, 256, 3)

        # -- Prepare Real Answer --
        mask_name = img_name # Filename is same
        mask_path = os.path.join(MASK_DIR, mask_name)
        mask = cv2.imread(mask_path, 0) # Grayscale

        # -- AI PREDICTION --
        pred_mask = model.predict(img_input, verbose=0)[0] # Output shape: (256, 256, 1)
        pred_mask = (pred_mask > 0.5).astype(np.uint8) # Convert decimals to 0 or 1 (Road vs No Road)

        # -- PLOTTING --
        # Column 1: Satellite Image
        axes[i, 0].imshow(img_rgb)
        axes[i, 0].set_title("Satellite Input")
        axes[i, 0].axis('off')

        # Column 2: Real Answer
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title("Real Road Map")
        axes[i, 1].axis('off')

        # Column 3: AI Prediction
        axes[i, 2].imshow(pred_mask, cmap='gray')
        axes[i, 2].set_title("AI Prediction 🧠")
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(SAVE_PATH)
    print(f"✅ Results saved to: {SAVE_PATH}")
    print("Go open that image and check the results!")

if __name__ == "__main__":
    predict_and_plot()