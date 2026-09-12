# PIL gives us Image (loading/saving) and ImageEnhance/ImageFilter for
# brightness and blur effects.
from PIL import Image, ImageEnhance, ImageFilter
import os

# CHANGE THIS to match one of your actual candidate image filenames.
SOURCE_IMAGE = "images/Andrew-Garfield-Award-Winning-Actor-PNG.png"

OUTPUT_FOLDER = "test_variations"


def make_variations():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load the original once, convert to RGB for consistency.
    original = Image.open(SOURCE_IMAGE).convert("RGB")

    # --- Resize (half dimensions) ---
    width, height = original.size
    resized = original.resize((width // 2, height // 2))
    resized.save(f"{OUTPUT_FOLDER}/resized.jpg")

    # --- Rotate 15 degrees ---
    # expand=True keeps the entire rotated image visible instead of
    # cropping corners that go outside the original frame.
    rotated = original.rotate(15, expand=True)
    rotated.save(f"{OUTPUT_FOLDER}/rotated.jpg")

    # --- Crop 20% off (10% from each side) ---
    crop_x = int(width * 0.10)
    crop_y = int(height * 0.10)
    cropped = original.crop((crop_x, crop_y, width - crop_x, height - crop_y))
    cropped.save(f"{OUTPUT_FOLDER}/cropped.jpg")

    # --- Brighten ---
    # ImageEnhance.Brightness(1.0) = unchanged, >1.0 = brighter, <1.0 = darker.
    brightened = ImageEnhance.Brightness(original).enhance(1.6)
    brightened.save(f"{OUTPUT_FOLDER}/brightened.jpg")

    # --- Darken ---
    darkened = ImageEnhance.Brightness(original).enhance(0.5)
    darkened.save(f"{OUTPUT_FOLDER}/darkened.jpg")

    # --- Blur ---
    # GaussianBlur(radius) - higher radius = more blur.
    blurred = original.filter(ImageFilter.GaussianBlur(radius=5))
    blurred.save(f"{OUTPUT_FOLDER}/blurred.jpg")

    # --- Low-quality JPEG compression ---
    # quality=10 is intentionally very aggressive to make the effect visible.
    original.save(f"{OUTPUT_FOLDER}/compressed.jpg", quality=10)

    print(f"Created variations in '{OUTPUT_FOLDER}/'")


if __name__ == "__main__":
    make_variations()