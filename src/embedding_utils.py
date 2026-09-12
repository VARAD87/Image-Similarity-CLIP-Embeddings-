# PIL is used to open and prepare images.
from PIL import Image

# torch is needed to disable gradient tracking during inference.
import torch

# numpy will hold our final embedding as a plain array (easier to work
# with than a raw PyTorch tensor for the similarity math we'll do later).
import numpy as np

# CLIPModel is the neural network; CLIPProcessor prepares raw images
# into the exact tensor format CLIP expects.
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"

# We load the model and processor ONCE, at import time, instead of inside
# the function. If embed_image() reloaded CLIP every call, processing
# 100 images would reload a ~600MB model 100 times - extremely slow.
print("Loading CLIP model...")
model = CLIPModel.from_pretrained(MODEL_NAME)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)
model.eval()
print("CLIP model ready.")


def embed_image(image_path):
    """
    Loads an image from image_path, runs it through CLIP, and returns
    its embedding as a normalized 1D NumPy array of shape (512,).
    """
    # Load the image and force RGB (CLIP expects 3-channel color images).
    image = Image.open(image_path).convert("RGB")

    # Preprocess: resize/normalize into the tensor format CLIP expects.
    inputs = processor(images=image, return_tensors="pt")

    # No gradient tracking needed - we're only doing inference, not training.
    with torch.no_grad():
        output = model.get_image_features(**inputs)

        # Handle the transformers 5.x wrapping behavior we discovered earlier.
        if hasattr(output, "image_embeds"):
            embedding = output.image_embeds
        elif hasattr(output, "pooler_output"):
            embedding = output.pooler_output
        else:
            embedding = output

    # Convert from a PyTorch tensor to a plain NumPy array, and drop the
    # extra "batch" dimension (shape goes from (1, 512) to just (512,)).
    embedding = embedding[0].numpy()

    # Normalize the vector to unit length (length = 1). This matters because
    # cosine similarity (Stage 7) measures the ANGLE between two vectors,
    # not their raw magnitude. Normalizing ensures that a slightly brighter
    # or larger version of the same image doesn't get penalized just for
    # having bigger raw numbers - only the *direction* of the vector
    # (i.e. its content/meaning) affects the similarity score.
    embedding = embedding / np.linalg.norm(embedding)

    return embedding


# Quick manual test - only runs if you execute this file directly.
if __name__ == "__main__":
    vec = embed_image("reference/reference.png")
    print("Embedding shape:", vec.shape)
    print("First 5 values:", vec[:5])
    print("Vector length (should be ~1.0):", np.linalg.norm(vec))