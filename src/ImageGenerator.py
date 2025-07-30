from PIL import Image
import ImagePixelExtractor
import Pixel
import numpy as np
import matplotlib.pyplot as plt

class ImageGenerator:
    def __init__(self, directory_path, max_x=512, max_y=512):
        """Initialize the ImageGenerator class."""
        self.directory_path = directory_path
        self.max_x = max_x
        self.max_y = max_y
        images = ImagePixelExtractor.get_full_image_array("../chest_Xray/test/PNEUMONIA/", self.max_x, self.max_y)
        self.image = np.array([[Pixel(x, y, images[x, y]) for y in range(max_y)] for x in range(max_x)])

    def generate_images_naive(self):
        self.generated_image = np.array([p.generate_pixel_value() for p in self.image.flatten()]).reshape(self.max_x, self.max_y)

    def show_img(self):
        plt.imshow(self.generated_image, cmap='gray')
        plt.colorbar()
        plt.title('Generated Image from Pixel Statistics')
        plt.show()
    