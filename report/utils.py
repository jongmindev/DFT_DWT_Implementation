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

import numpy as np


def morlet(omega_x, omega_y, epsilon=1, sigma=1, omega_0=2):
    return np.exp(-sigma**2 * ((omega_x - omega_0)**2 + (epsilon * omega_y)**2) / 2)


def mexh(omega_x, omega_y, sigma_y=1, sigma_x=1, order=2):
    return -(2 * np.pi) * (omega_x**2 + omega_y**2)**(order / 2) * \
           np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


def gaus(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1):
    return (1j * omega_x)**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


def gaus_2(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1):
    return (1j * (omega_x + 1j * omega_y))**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


def gaus_3(omega_x, omega_y, sigma_y=1, sigma_x=1, order=1, b=1, a=1):
    return (1j * (a * omega_x + b * 1j * omega_y))**order * np.exp(-((sigma_x * omega_x)**2 + (sigma_y * omega_y)**2) / 2)


def cauchy(omega_x, omega_y, cone_angle=np.pi / 6, sigma=1, l=4, m=4):
    dot1 = np.sin(cone_angle) * omega_x + np.cos(cone_angle) * omega_y
    dot2 = -np.sin(cone_angle) * omega_x + np.cos(cone_angle) * omega_y
    coef = (dot1 ** l) * (dot2 ** m)

    k0 = (l + m) ** 0.5 * (sigma - 1) / sigma
    rad2 = 0.5 * sigma * ((omega_x - k0)**2 + omega_y**2)
    pond = np.tan(cone_angle) * omega_x > abs(omega_y)
    wft = pond * coef * np.exp(-rad2)

    return wft


def dog(omega_x, omega_y, alpha=1.25):
    m = (omega_x**2 + omega_y**2) / 2
    wft = -np.exp(-m) + np.exp(-alpha**2 * m)

    return wft


c_wavelets = dict(
    morlet=morlet,
    mexh=mexh,
    gaus=gaus,
    gaus_2=gaus_2,
    gaus_3=gaus_3,
    cauchy=cauchy,
    dog=dog
)