import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from safetensors.torch import load_file
from typing import Dict, List
from PIL import Image
import shutil
import argparse
import os

model_path = "model/model.safetensors"

model = ViTForImageClassification.from_pretrained("model")
state_dict = load_file(model_path)
model.load_state_dict(state_dict, strict=False)
model.eval()

processor = ViTImageProcessor.from_pretrained("model")

def predict(image_path: str):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = logits.argmax(-1).item()
    id2label = {0: "cute", 1: "erotic", 2: "sexy"}
    return id2label[predicted_class]

def move_images(src_folder: str, dest_folder: str):
    files = [f for f in os.listdir(src_folder) if f != ".DS_Store" and f != "files.json"]
    for file in files:
        src = os.path.join(src_folder, file)
        if os.path.isdir(src):
            continue
        dest = os.path.join(dest_folder, file)
        shutil.move(src, dest)

def process_dir(dir: str):
    cute_dir = os.path.join(dir, "cute")
    sexy_dir = os.path.join(dir, "sexy")
    erotic_dir = os.path.join(dir, "erotic")
    process_dir = os.path.join(dir, "original")

    os.makedirs(cute_dir, exist_ok=True)
    os.makedirs(sexy_dir, exist_ok=True)
    os.makedirs(erotic_dir, exist_ok=True)
    os.makedirs(process_dir, exist_ok=True)

    move_images(dir, process_dir)

    files = [f for f in os.listdir(process_dir) if os.path.isfile(os.path.join(process_dir, f)) and f != ".DS_Store"]

    obj: Dict[str, List[str]] = {}
    for file in files:
        id = file if "_s" in file else file.split("_")[0]
        if id in obj:
            obj[id].append(os.path.join(process_dir, file))
        else:
            obj[id] = [os.path.join(process_dir, file)]

    i = 0
    for key, images in obj.items():
        rating = predict(images[0])

        for image in images:
            if rating == "cute":
                shutil.move(image, os.path.join(cute_dir, os.path.basename(image)))
            elif rating == "sexy":
                shutil.move(image, os.path.join(sexy_dir, os.path.basename(image)))
            elif rating == "erotic":
                shutil.move(image, os.path.join(erotic_dir, os.path.basename(image)))

        print(f"{i + 1} / {len(obj)}")
        i += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rate Images")
    parser.add_argument("folderpath")
    args = parser.parse_args()
    process_dir(args.folderpath)