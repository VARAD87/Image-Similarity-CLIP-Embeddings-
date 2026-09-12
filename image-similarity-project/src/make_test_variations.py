
from PIL import Image, ImageEnhance, ImageFilter
import os

SOURCE_IMAGE = "images/Andrew-Garfield-Award-Winning-Actor-PNG.png"

OUTPUT_FOLDER = "test_variations"


def make_variations():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    original = Image.open(SOURCE_IMAGE).convert("RGB")

    # --- Resize (half dimensions) ---
    width, height = original.size
    resized = original.resize((width // 2, height // 2))
    resized.save(f"{OUTPUT_FOLDER}/resized.jpg")

    # --- Rotate 15 degrees ---

    rotated = original.rotate(15, expand=True)
    rotated.save(f"{OUTPUT_FOLDER}/rotated.jpg")

    # --- Crop 20% off (10% from each side) ---
    crop_x = int(width * 0.10)
    crop_y = int(height * 0.10)
    cropped = original.crop((crop_x, crop_y, width - crop_x, height - crop_y))
    cropped.save(f"{OUTPUT_FOLDER}/cropped.jpg")

    # --- Brighten ---

    brightened = ImageEnhance.Brightness(original).enhance(1.6)
    brightened.save(f"{OUTPUT_FOLDER}/brightened.jpg")

    # --- Darken ---
    darkened = ImageEnhance.Brightness(original).enhance(0.5)
    darkened.save(f"{OUTPUT_FOLDER}/darkened.jpg")

    # --- Blur ---
   
    blurred = original.filter(ImageFilter.GaussianBlur(radius=5))
    blurred.save(f"{OUTPUT_FOLDER}/blurred.jpg")

    # --- Low-quality JPEG compression ---
    
    original.save(f"{OUTPUT_FOLDER}/compressed.jpg", quality=10)

    print(f"Created variations in '{OUTPUT_FOLDER}/'")


if __name__ == "__main__":
    make_variations()
