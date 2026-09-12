from PIL import Image
import torch
import numpy as np
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print("CLIP model ready.")


def embed_image(image_path):

    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        output = model.get_image_features(**inputs)

        if hasattr(output, "image_embeds"):
            embedding = output.image_embeds
        elif hasattr(output, "pooler_output"):
            embedding = output.pooler_output
        else:
            embedding = output

    embedding = embedding[0].numpy()

    embedding = embedding / np.linalg.norm(embedding)

    return embedding

if __name__ == "__main__":
    vec = embed_image("reference/reference.png")
    print("Embedding shape:", vec.shape)
    print("First 5 values:", vec[:5])
    print("Vector length (should be ~1.0):", np.linalg.norm(vec))
