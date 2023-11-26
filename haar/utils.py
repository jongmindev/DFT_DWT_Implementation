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
