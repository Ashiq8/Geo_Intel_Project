import os
import cv2
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
RAW_IMAGE_DIR = "dataset/archive (1)/road_segmentation_ideal/training/input"
RAW_MASK_DIR  = "dataset/archive (1)/road_segmentation_ideal/training/output"

SAVE_IMG_DIR  = "processed_data/train_images/"
SAVE_MASK_DIR = "processed_data/train_masks/"

PATCH_SIZE = 256
STEP = 256  # No overlap

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def process_data():
    print(f"🚀 Starting to process images (Silent Mode)...")
    
    create_dir(SAVE_IMG_DIR)
    create_dir(SAVE_MASK_DIR)
    
    image_list = sorted(os.listdir(RAW_IMAGE_DIR))
    print(f"📂 Found {len(image_list)} images. Chopping them now...")
    
    count = 0
    
    for img_name in tqdm(image_list):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')): 
            continue
        
        # 1. Read Input Image
        img_path = os.path.join(RAW_IMAGE_DIR, img_name)
        img = cv2.imread(img_path)
        
        if img is None: continue # Skip corrupt images

        # 2. Find Matching Mask (Smart Search)
        # We check extensions silently to avoid OpenCV warnings
        mask = None
        prefix = img_name.rsplit('.', 1)[0] # remove extension
        
        # Try finding the mask with different extensions
        possible_extensions = ['.png', '.tif', '.tiff', '.jpg']
        
        for ext in possible_extensions:
            try_path = os.path.join(RAW_MASK_DIR, prefix + ext)
            if os.path.exists(try_path):
                mask = cv2.imread(try_path, 0) # Read grayscale
                break
        
        if mask is None:
            # If still not found, try original name exactly
            try_path = os.path.join(RAW_MASK_DIR, img_name)
            if os.path.exists(try_path):
                mask = cv2.imread(try_path, 0)

        if mask is None:
            continue # Skip if no mask found (don't crash)

        # 3. Crop into patches
        h, w, _ = img.shape
        for i in range(0, h, STEP):
            for j in range(0, w, STEP):
                if i + PATCH_SIZE > h or j + PATCH_SIZE > w: continue
                
                img_patch = img[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                mask_patch = mask[i:i+PATCH_SIZE, j:j+PATCH_SIZE]
                
                # Check if patch has any road (White pixels)
                # We relax the check: even if it has a LITTLE road, keep it.
                if np.sum(mask_patch) > 0: 
                    save_name = f"{prefix}_{i}_{j}.png"
                    cv2.imwrite(os.path.join(SAVE_IMG_DIR, save_name), img_patch)
                    cv2.imwrite(os.path.join(SAVE_MASK_DIR, save_name), mask_patch)
                    count += 1
                    
    print(f"✅ Finished! Created {count} clean patches.")

if __name__ == "__main__":
    if os.path.exists(RAW_IMAGE_DIR):
        process_data()
    else:
        print(f"⚠️ PATH ERROR: Could not find {RAW_IMAGE_DIR}")