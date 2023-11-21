from typing import Union
import numpy as np
from tqdm import tqdm
import utils


class DFT:
    def __init__(self, image_or_path: Union[np.ndarray, str]) -> None:
        if isinstance(image_or_path, np.ndarray):
            self.image = image_or_path
        elif isinstance(image_or_path, str):
            self.image = utils.load_image(image_or_path)
        else:
            raise TypeError("Invalid type : image_or_path")
        
        try:
            self.image_grayscaled = utils.grayscaling(self.image)
        except AssertionError:
            self.image_grayscaled = self.image
    
    def imshow(self, original: bool = True):
        if original:
            utils.imshow(self.image, "Original image", False)
        else:
            utils.imshow(self.image_grayscaled, "Grayscaled image", True)


class NaiveDFT(DFT):
    def __init__(self, image_or_path: Union[np.ndarray, str]) -> None:
        super().__init__(image_or_path)
    
    def dft2d(self) -> np.ndarray:
        W, H = self.image_grayscaled.shape
        out = np.zeros_like(self.image_grayscaled, dtype='complex_')
        for (w, h), _ in tqdm(np.ndenumerate(self.image_grayscaled)):
            val = 0
            for (ww, hh), pixel in np.ndenumerate(self.image_grayscaled):
                e = np.exp(-1j * 2 * np.pi * (w * ww / W + h * hh / H))
                val += pixel * e
            out[w, h] = val
        self.dft2d_array = out
        return self.dft2d_array
    
    def idft2d(self) -> np.ndarray:
        W, H = self.dft2_array.shape
        image = np.zeros_like(self.dft2_array, dtype='float64')
        for (w, h), _ in tqdm(np.ndenumerate(self.dft2_array)):
            pixel = 0
            for (ww, hh), val in np.ndenumerate(self.dft2_array):
                e = np.exp(1j * 2 * np.pi * (w * ww / W + h * hh / H))
                pixel += val * e
            image[w, h] = pixel.real
        image = image / W / H
        self.restored_image = image
        return self.restored_image
    
class VectorizedDFT(DFT):
    def __init__(self, image_or_path: Union[np.ndarray, str]) -> None:
        super().__init__(image_or_path)
    
    def dft2d_vecvec(self) -> np.ndarray:
        W, H = self.image_grayscaled.shape
        wo = (np.outer(np.arange(W), np.arange(W)) * (2 * np.pi / W)) * -1j
        ho = (np.outer(np.arange(H), np.arange(H)) * (2 * np.pi / H)) * -1j
        ewo = np.exp(wo)
        eho = np.exp(ho)
        eo = np.multiply.outer(ewo, eho).transpose(0, 2, 1, 3)
        pixelo = np.broadcast_to(self.image_grayscaled, shape=(W, H, W, H))
        valo = pixelo * eo
        out = np.sum(valo, axis=(2, 3))
        self.dft2d_array = out
        return self.dft2d_array
    
    def dft2d_vec(self) -> np.ndarray:
        W, H = self.image_grayscaled.shape
        out = np.zeros_like(self.image_grayscaled, dtype='complex_')
        for (w, h), _ in tqdm(np.ndenumerate(self.image_grayscaled)):
            wo = np.arange(W) * ((2 * w * np.pi) / W) * -1j
            ho = np.arange(H) * ((2 * h * np.pi) / H) * -1j
            ewo = np.exp(wo)
            eho = np.exp(ho)
            eo = np.multiply.outer(ewo, eho)
            val = self.image_grayscaled * eo
            out[w, h] = np.sum(val)
        self.dft2d_array = out
        return self.dft2d_array