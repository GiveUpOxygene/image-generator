from PIL import Image
import os
import numpy as np

def open_image(file_path):
    """
    Opens an image file and returns the Image object.

    :param file_path: Path to the image file.
    :return: PIL.Image.Image object.
    """
    try:
        image = Image.open(file_path)
        print(f"Image opened successfully: {file_path}")
        return image
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    

def extract_pixel_values_from_directory(directory_path, x, y, max_x=512, max_y=512):
    """
    Extracts the greyscale value of the pixel at (x, y) from each image in the directory.

    :param directory_path: Path to the directory containing images.
    :param x: X-coordinate of the pixel.
    :param y: Y-coordinate of the pixel.
    :param max_x: Maximum width of the images.
    :param max_y: Maximum height of the images.
    :return: np.array of greyscale pixel values.
    """
    pixel_values = []
    print(f"Extracting pixel values from directory: {directory_path} at coordinates ({x}, {y})")
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.isfile(file_path):
            image = open_image(file_path)
            print(f"Processing file: {file_name}")
            if image is not None:
                image = np.array(image)
                image = Image.fromarray(image).resize((max_x, max_y))
                if image.size[0] >= max_x and image.size[1] >= max_y:
                    image = image.convert("L")  # Convert to greyscale
                    pixel_value = image.getpixel((x, y))
                    pixel_values.append(pixel_value)
                else:
                    print(f"Image {file_name} does not meet the size requirements.")
            else:
                print(f"Skipping file {file_name} as it could not be opened.")
    
    return np.array(pixel_values)