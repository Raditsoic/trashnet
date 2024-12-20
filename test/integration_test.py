import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import TrashnetModel
import torch
import os
from PIL import Image
from torchvision import transforms
import json

labels = {
    0: "cardboard",
    1: "glass",
    2: "metal",
    3: "paper",
    4: "plastic",
    5: "trash"
}


def load_model(version, model, base_dir="models"):
    model_dir = os.path.join(base_dir, f"model_{version}")
    model_path = os.path.join(model_dir, f"model_{version}.pth")
    config_path = os.path.join(model_dir, "config.json")

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    with open(config_path, "r") as f:
        config = json.load(f)

    print(f"Loaded model version {version} with config: {config}")
    return model, config

model = TrashnetModel(num_classes=6)  

# Load Model
model_version = "1.1" 
model, config = load_model(model_version, model)

# Image preprocessing to match training
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_folder = "test/images" 
test_images = os.listdir(test_folder)

model.eval()
for image_name in test_images:
    image_path = os.path.join(test_folder, image_name)
    image = Image.open(image_path).convert("RGB")  

    # Preprocess
    input_tensor = transform(image).unsqueeze(0) 

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted_class = torch.max(output, 1)

    print(f"Image: {image_name}, Predicted Class: {labels[predicted_class.item()]}")
