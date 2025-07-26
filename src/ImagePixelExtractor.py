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
        # print(f"Image opened successfully: {file_path}")
        return image
    except Exception as e:
        print(f"Error opening image: {e}")
        return None
    
def pixel_value_from_image(image, x, y, max_x=512, max_y=512):
    """
    Extracts the greyscale value of the pixel at (x, y) from the given image.

    :param image: PIL.Image.Image object.
    :param x: X-coordinate of the pixel.
    :param y: Y-coordinate of the pixel.
    :return: Greyscale pixel value at (x, y).
    """
    if image is not None:
        image = image.convert("L").resize((max_x, max_y))
        if x < 0 or y < 0 or x >= image.size[0] or y >= image.size[1]:
            print(f"Coordinates ({x}, {y}) are out of bounds for image size {image.size}.")
            return None
        return image.getpixel((x, y))
    else:
        print("Image is None, cannot extract pixel value.")
        return None

def extract_pixel_values_from_directory(directory_path, x=-1, y=-1, max_x=512, max_y=512, low_mem=True):
    if (not low_mem):
        return _extract_pixel_values_from_directory_high_mem(directory_path, max_x, max_y)
    elif (low_mem and x != -1 and y !=-1):
        return _extract_pixel_values_from_directory_low_mem(directory_path, x, y, max_x, max_y)
    else:
        raise ValueError("Invalid parameters: If low_mem is True, x and y must be specified. If low_mem is False, x and y should not be specified.")    


def _extract_pixel_values_from_directory_low_mem(directory_path, x, y, max_x=512, max_y=512):
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
            if image is not None:
                pixel_value = pixel_value_from_image(image, x, y, max_x, max_y)
                if pixel_value is not None:
                    pixel_values.append(pixel_value)
            else:
                print(f"Skipping file {file_name} as it could not be opened.")
    
    return np.array(pixel_values)


def _extract_pixel_values_from_directory_high_mem(directory_path, max_x=512, max_y=512):
    """
    Loads all images from the directory into memory, resizes them, and extracts the greyscale values
    for all pixels, returning a 3D array of pixel values.

    :param directory_path: Path to the directory containing images.
    :param x: X-coordinate of the pixel (ignored in this function).
    :param y: Y-coordinate of the pixel (ignored in this function).
    :param max_x: Maximum width of the images.
    :param max_y: Maximum height of the images.
    :return: np.array of shape (num_images, max_y, max_x) containing greyscale pixel values.
    """
    images = []
    print(f"Loading all images from directory: {directory_path}")
    for file_name in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file_name)
        
        if os.path.isfile(file_path):
            image = open_image(file_path)
            if image is not None:
                image = image.convert("L").resize((max_x, max_y))
                images.append(np.array(image))
            else:
                print(f"Skipping file {file_name} as it could not be opened.")
    
    return np.array(images)


def get_full_image_array(directory_path, max_x=512, max_y=512):
    tab = extract_pixel_values_from_directory(directory_path, max_x, max_y, low_mem=False)
    return tab.transpose(1,2,0)