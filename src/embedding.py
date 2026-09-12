import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model... (this may take a minute the first time)")

model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded.")

image = Image.open("reference/reference.png").convert("RGB")

inputs = processor(images=image, return_tensors="pt")


with torch.no_grad():
    output = model.get_image_features(**inputs)

    if hasattr(output, "image_embeds"):
    
        image_features = output.image_embeds

    elif hasattr(output, "pooler_output"):

        image_features = output.pooler_output

    else:
        image_features = output


print("Embedding shape:", image_features.shape)
