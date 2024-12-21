import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.utils import load_img, img_to_array
import re

# Labels for classification
labels = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

# Preprocess image to match model input
def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))  # Resize
    img_array = img_to_array(img) / 255.0              # Rescale
    return np.expand_dims(img_array, axis=0)      

# Load the latest model
def get_latest_version(base_dir="models"):
    model_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    version_pattern = re.compile(r'model_(\d+\.\d+)')
    
    versions = []
    for model_dir in model_dirs:
        match = version_pattern.match(model_dir)
        if match:
            versions.append(match.group(1))
    
    if not versions:
        raise ValueError(f"No valid model versions found in {base_dir}.")
    
    latest_version = sorted(versions, key=lambda v: [int(x) for x in v.split('.')])[-1]
    return latest_version

base_model_dir = "models"
latest_version = get_latest_version(base_model_dir)

model_path = os.path.join(base_model_dir, f"model_{latest_version}", f"model_{latest_version}.keras")
model = tf.keras.models.load_model(model_path)
print(f"Loaded model version: {latest_version}")

# Dummy data for inference
test_folder = "test/images"
test_images = os.listdir(test_folder)

# Inference on test images
for image_name in test_images:
    image_path = os.path.join(test_folder, image_name)
    input_tensor = preprocess_image(image_path)  

    # Inference
    predictions = model.predict(input_tensor)
    predicted_class = np.argmax(predictions, axis=1)[0]

    print(f"Image: {image_name}, Predicted Class: {labels[predicted_class]}")
