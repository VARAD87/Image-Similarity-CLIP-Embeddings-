# PIL is used to open and prepare the image.
from PIL import Image

# torch is needed to disable gradient tracking during inference (explained below).
import torch

# CLIPModel is the actual neural network. CLIPProcessor prepares
# raw images/text into the exact tensor format CLIP expects.
from transformers import CLIPModel, CLIPProcessor

# This is the name of the pretrained CLIP model we're downloading from
# Hugging Face's model hub. It will download automatically the first time.
MODEL_NAME = "openai/clip-vit-base-patch32"

print("Loading CLIP model... (this may take a minute the first time)")

# Load the pretrained model weights.
model = CLIPModel.from_pretrained(MODEL_NAME)

# Load the matching processor, which knows how CLIP expects images
# to be resized/normalized before being fed into the model.
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# Put the model into "evaluation mode." This disables training-specific
# behavior (like dropout) that we don't want during inference.
model.eval()

print("Model loaded.")

# Load our reference image and make sure it's in RGB
# (CLIP expects 3-channel color images, some PNGs are RGBA or grayscale).
image = Image.open("reference/reference.png").convert("RGB")

# The processor converts the PIL image into a tensor: resizing, normalizing
# pixel values, and packaging it into the format CLIP's model expects.
# return_tensors="pt" means "give me PyTorch tensors."
inputs = processor(images=image, return_tensors="pt")

# torch.no_grad() tells PyTorch: "don't track operations for backpropagation."
# Backpropagation is only needed during training, to compute how to adjust
# the model's weights. Since we're only doing inference (not training),
# skipping this tracking makes computation faster and uses less memory.
with torch.no_grad():
    output = model.get_image_features(**inputs)

    # Handle differences across transformers versions:
    if hasattr(output, "image_embeds"):
        # Some versions return the final projected embedding directly
        # under this attribute name.
        image_features = output.image_embeds

    elif hasattr(output, "pooler_output"):
        # transformers 5.x (your version): get_image_features() wraps its
        # result in a BaseModelOutputWithPooling object instead of returning
        # a plain tensor. Its pooler_output is already the final projected
        # 512-d embedding (confirmed by its shape being 512, not the 768-d
        # raw vision hidden size) - so we use it directly, no extra
        # projection needed.
        image_features = output.pooler_output

    else:
        # Older transformers versions returned a plain tensor directly.
        image_features = output

# This line must NOT be indented inside the "with" block above -
# it runs after the embedding has been extracted.
print("Embedding shape:", image_features.shape)