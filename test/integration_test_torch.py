import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import TrashnetModel
import torch
import os
from PIL import Image
from torchvision import transforms
import json
import re

labels = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}

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

def load_model(version, model, base_dir="models"):
    model_dir = os.path.join(base_dir, f"model_{version}")
    model_path = os.path.join(model_dir, f"model_{version}.pth")
    config_path = os.path.join(model_dir, "config.json")

    # Load Weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load Config
    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Loaded model version {version} with config: {config}")
    return model, config

# Test only latest model ver
base_model_dir = "models"
latest_version = get_latest_version(base_model_dir)

model = TrashnetModel(num_classes=6)
model, config = load_model(latest_version, model, base_model_dir)

# Image preprocessing to match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Dummy data for inference
test_folder = "test/images"
test_images = os.listdir(test_folder)

# Inference on test images
model.eval()
for image_name in test_images:
    image_path = os.path.join(test_folder, image_name)
    image = Image.open(image_path).convert("RGB")  

    # Preprocess data
    input_tensor = transform(image).unsqueeze(0) 

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    print(f"Image: {image_name}, Predicted Class: {labels[predicted_class.item()]}")