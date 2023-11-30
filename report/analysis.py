from tqdm import tqdm
import utils
import transform


class Visualization:
    def __init__(self, transformer: transform.Transform) -> None:
        self.transformer = transformer
        self.reconstructed_images = None
    
    def compress_by_ratio(self):
        retention_percentages = {
            "100%": 100,
            "50%": 50,
            "10%": 90,
            "1%": 99,
            "0.1%": 99.9,
            "0.05%": 99.95,
            "0.01%": 99.99,
            # "0.008%": 99.992,
            # "0.006%": 99.994,
            # "0.004%": 99.996,
            # "0.002%": 99.998,
            # "0.001%": 99.999,
            # "0.0008%": 99.9992,
            # "0.0006%": 99.9994,
            # "0.0004%": 99.9996,
            # "0.0002%": 99.9998,
            # "0.0001%": 99.9999,
            # "0.00008%": 99.99992,
            # "0.00006%": 99.99994,
            # "0.00004%": 99.99996,
            # "0.00002%": 99.99998,
            # "0.00001%": 99.99999,
        }
        images = {"original": self.transformer.loader.image_grayscaled}

        for percentage, value in tqdm(retention_percentages.items()):
            images[percentage] = self.transformer.compress_image(value)

        self.reconstructed_images = images

    def compare_the_whole_by_compression_ratio(self, **kwargs):
        if self.reconstructed_images:
            images = self.reconstructed_images
        else:
            self.compress_by_ratio()
            images = self.reconstructed_images
        utils.plot_images(images, **kwargs)
    
    def compare_the_waffle_by_compression_ratio(self, **kwargs):
        if self.reconstructed_images:
            images = self.reconstructed_images
        else:
            self.compress_by_ratio()
            images = self.reconstructed_images
        for k, img in images.items():
            images[k] = img[550:1150,3600:4200]
        utils.plot_images(images, **kwargs)