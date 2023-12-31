from tqdm import tqdm
import utils
import transform
import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, transformer: transform.Transform, fewer = True) -> None:
        self.transformer = transformer
        self.reconstructed_images = None
        self.fewer = fewer
    
    def compress_by_ratio(self):
        fewer_retention_percentages = {
            "100%": 0,
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
        more_retention_percentages = {
            "100%": 0,
            "50%": 50,
            "10%": 90,
            "1%": 99,
            "0.1%": 99.9,
            "0.05%": 99.95,
            "0.01%": 99.99,
            "0.008%": 99.992,
            "0.006%": 99.994,
            "0.004%": 99.996,
            "0.002%": 99.998,
            "0.001%": 99.999,
            "0.0008%": 99.9992,
            "0.0006%": 99.9994,
            "0.0004%": 99.9996,
            "0.0002%": 99.9998,
            "0.0001%": 99.9999,
            "0.00008%": 99.99992,
            "0.00006%": 99.99994,
            "0.00004%": 99.99996,
            "0.00002%": 99.99998,
            "0.00001%": 99.99999,
        }
        retention_percentages = fewer_retention_percentages if self.fewer else more_retention_percentages
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
        images_cropped = {}
        if self.reconstructed_images:
            images = self.reconstructed_images
        else:
            self.compress_by_ratio()
            images = self.reconstructed_images
        for k, img in images.items():
            images_cropped[k] = img[550:1150,3600:4200]
        utils.plot_images(images_cropped, **kwargs)
    
    def plot_rmse_by_compression_ratio(self,plot_onto_same_graph = False, **kwargs):
        original = self.transformer.loader.image_grayscaled
        if self.reconstructed_images : 
            images = self.reconstructed_images
        else:
            self.compress_by_ratio()
            images = self.reconstructed_images
        rmse_list = []
        x_values = []
        for percentage, img in images.items():
            if percentage == "original":
                pass
            else : 
                rmse_list.append(utils.calculate_rmse(original, img))
                x_values.append(0.01*float(percentage.rstrip('%')))  # Remove the percentage sign 
        
        plt.plot(x_values, rmse_list, **kwargs)
        if not plot_onto_same_graph:
            plt.xlabel("Compression Rate")
            plt.ylabel("RMSE")
            plt.title("Reconstruction Error Rate vs Compression Rate")
            plt.legend()
            plt.xscale("log")
            plt.show()
        
    
    def plot_histogram(self, percentages_to_plot,xrange = [0,1], yrange = [0,300000], **kwargs):
        original = self.transformer.loader.image_grayscaled
        if self.reconstructed_images:
            images = self.reconstructed_images
        else:
            self.compress_by_ratio()
            images = self.reconstructed_images

        #plotting
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(utils.flatten_rescale(original), bins=256, range=[0, 1], density=False, alpha=1, color = "grey", label="Original Image")
        for percentqage in percentages_to_plot:
            pixel_vals = utils.flatten_rescale(images[percentqage])
            ax.hist(pixel_vals, bins=256, range=[0, 1], density=False, alpha=0.3, label=f"Reconstructed Image ({percentqage})")
        ax.set_title('Histogram Comparison')
        ax.set_xlabel('Pixel Value (normalized)')
        ax.set_ylabel('Frequency')
        ax.legend()
        plt.xlim(xrange[0], xrange[1])
        plt.ylim(yrange[0], yrange[1])
        plt.show()
