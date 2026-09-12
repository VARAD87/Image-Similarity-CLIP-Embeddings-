
from PIL import Image


def show_image_info(image_path):
    image = Image.open(image_path)

    width, height = image.size

    mode = image.mode

    print(f"Filename: {image_path}")
    print(f"Width: {width}")
    print(f"Height: {height}")
    print(f"Color mode: {mode}")


if __name__ == "__main__":
    show_image_info("reference/reference.png")
