import os
import numpy as np
import tensorflow as tf
from model import build_unet
import cv2
from glob import glob
from sklearn.model_selection import train_test_split

# --- SETTINGS ---
IMG_SIZE = 256
BATCH_SIZE = 8   # If you get "OOM" (Out of Memory) error, change this to 4
EPOCHS = 10      # Start with 10. Increase to 30 later for better results.
LR = 1e-4        # Learning Rate

# --- DATA LOADING FUNCTIONS ---
def load_paths(path):
    images = sorted(glob(os.path.join(path, "train_images/*")))
    masks = sorted(glob(os.path.join(path, "train_masks/*")))
    return images, masks

def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = x / 255.0  # Normalize to 0-1
    return x.astype(np.float32)

def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = x / 255.0  # Normalize to 0-1
    x = np.expand_dims(x, axis=-1) # Make it (256, 256, 1)
    return x.astype(np.float32)

def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([IMG_SIZE, IMG_SIZE, 3])
    y.set_shape([IMG_SIZE, IMG_SIZE, 1])
    return x, y

def tf_dataset(x, y, batch=8):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.map(tf_parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Load Data
    print("⏳ Loading Data Paths...")
    # Pointing to the folder we just filled with preprocess.py
    images, masks = load_paths("processed_data/")
    print(f"📂 Found {len(images)} images and {len(masks)} masks.")

    # 2. Split Data (80% Train, 20% Validation)
    train_x, valid_x, train_y, valid_y = train_test_split(images, masks, test_size=0.2, random_state=42)

    # 3. Create Data Pipelines
    train_dataset = tf_dataset(train_x, train_y, batch=BATCH_SIZE)
    valid_dataset = tf_dataset(valid_x, valid_y, batch=BATCH_SIZE)

    # 4. Build & Compile Model
    print("🔧 Building U-Net...")
    model = build_unet()
    model.compile(optimizer=tf.keras.optimizers.Adam(LR), 
                  loss='binary_crossentropy', 
                  metrics=['accuracy'])

    # 5. Train
    print("🚀 Starting Training (This will take time)...")
    model.fit(train_dataset, 
              epochs=EPOCHS, 
              validation_data=valid_dataset)

    # 6. Save Model
    model.save('geo_intel_model.h5')
    print("✅ Training Finished! Model saved as 'geo_intel_model.h5'")