from typing import Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread


def imshow(image: np.ndarray, title: str, grayscale: bool = False):
    plt.title(title)
    if grayscale:
        plt.imshow(image, cmap='gray')
    else:
        plt.imshow(image)
    plt.axis('off')
    plt.show()

def load_image(path: str) -> np.ndarray:
    return imread(path)

def grayscaling(image: np.ndarray) -> np.ndarray:
    assert isinstance(image, np.ndarray)
    assert (len(image.shape) == 3) and (image.shape[-1] == 3)
    return np.mean(image, axis=-1)

def plot_images(images: dict, figsize=(24, 20), grid_width: int = 4, line_in_center = False):
    # Create a figure
    grid_height = len(images) // grid_width + ((len(images) % grid_width) > 0)
    fig, axs = plt.subplots(grid_height, grid_width, figsize=figsize)

    # Plot images in the figure
    for i, (percentage, img) in enumerate(images.items()):
        # Remove the percentage sign from the string
        row = i // grid_width
        col = i % grid_width
        axs[row, col].imshow(img, cmap='gray')
        axs[row, col].set_title(percentage)
        axs[row, col].axis('off')
        if line_in_center:
            axs[row, col].axvline(x=img.shape[1] // 2 , color='red', linestyle='--', linewidth=3)
            axs[row, col].axhline(y=img.shape[0] // 2 , color='red', linestyle='--', linewidth=3)
    
    # Remove empty subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.2)  # Adjust the hspace parameter to change the space between rows
    plt.axis('off')

    plt.show()


class ImageLoader:
    def __init__(self, image_or_path: Union[np.ndarray, str]) -> None:
        if isinstance(image_or_path, np.ndarray):
            self.image = image_or_path
        elif isinstance(image_or_path, str):
            self.image = load_image(image_or_path)
        else:
            raise TypeError("Invalid type : image_or_path")
        
        try:
            self.image_grayscaled = grayscaling(self.image)
        except AssertionError:
            self.image_grayscaled = self.image
    
    def imshow(self, original: bool = True):
        if original:
            imshow(self.image, "Original image", False)
        else:
            imshow(self.image_grayscaled, "Grayscaled image", True)