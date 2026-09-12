# We need the Image class from Pillow to open and inspect images.
from PIL import Image


def show_image_info(image_path):
    """
    Loads an image from the given path and prints basic information:
    filename, width, height, and color mode.
    """
    # Image.open() reads the image file from disk into a Python object
    # we can inspect and manipulate. It does not load all pixel data
    # immediately — Pillow is lazy about that until you actually need it.
    image = Image.open(image_path)

    # image.size returns a tuple (width, height) in pixels.
    width, height = image.size

    # image.mode tells us how color is stored, e.g. "RGB", "L" (grayscale), "RGBA".
    mode = image.mode

    print(f"Filename: {image_path}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Color mode: {mode}")


# This block only runs when we execute this file directly
# (not when it's imported by another file later).
if __name__ == "__main__":
    show_image_info("reference/reference.png")