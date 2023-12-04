from tqdm import tqdm
import utils
import transform
import numpy as np
import matplotlib.pyplot as plt


class Visualization:
    def __init__(self, transformer: transform.Transform, dwt=True, fewer = True) -> None:
        self.transformer = transformer
        self.reconstructed_images = None
        self.fewer = fewer
    
    def compress_by_ratio(self, dwt=True):
        fewer_retention_percentages = {
            "100%": 0,
            "50%": 50,
            "10%": 90,
            "1%": 99,
            "0.1%": 99.9,
            "0.05%": 99.95,
            "0.01%": 99.99
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
            if dwt:
                images[percentage] = self.transformer.compress_image(value)
            else: # CWT
                images[percentage] = self.transformer.cwt_compress_image(value)

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
            plt.xscale("log")
            plt.xlabel("1 - Compression Rate")
            plt.ylabel("RMSE")
            plt.title("Reconstruction Error Rate vs Compression Rate")
            plt.legend()
            plt.show()
    
    def plot_histogram(self, percentages_to_plot, xrange=[0, 1], yrange=[0, 300000], second_image=None, **kwargs):
        original = self.transformer.loader.image_grayscaled
        if self.reconstructed_images:
            images = self.reconstructed_images
        else:
            self.compress_by_ratio()
            images = self.reconstructed_images

        if second_image:
            if second_image.reconstructed_images:
                second_image_recon = second_image.reconstructed_images
            else:
                second_image.compress_by_ratio()
                second_image_recon = second_image.reconstructed_images

        # Plotting
        if second_image:
            fig , (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        else : 
            fig, ax1 = plt.subplots(1, 1, figsize=(6, 4))

        # Plot histogram for original image
        ax1.hist(utils.flatten_rescale(original), bins=256, range=[0, 1], density=False, alpha=1, color="grey",
                 label="Original Image")
        # Plot histogram for first reconstructed image
        for percentage in percentages_to_plot:
            pixel_vals = utils.flatten_rescale(images[percentage])
            ax1.hist(pixel_vals, bins=256, range=[0, 1], density=False, alpha=0.3, label=f"Reconstruction ({percentage})")
        if 'title_1' in kwargs:
            if kwargs['title_1']:
                ax1.set_title(kwargs['title_1'])
        else:
            ax1.set_title('Histogram Comparison')
        ax1.set_xlabel('Pixel Intensity')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.set_xlim(xrange[0], xrange[1])
        ax1.set_ylim(yrange[0], yrange[1])

        # Plot histogram for second reconstructed image if provided
        if second_image:
            # Plot histogram for original image
            ax2.hist(utils.flatten_rescale(original), bins=256, range=[0, 1], density=False, alpha=1, color="grey",
                 label="Original Image")
            for percentage in percentages_to_plot:
                pixel_vals = utils.flatten_rescale(second_image_recon[percentage])
                ax2.hist(pixel_vals, bins=256, range=[0, 1], density=False, alpha=0.3,
                         label=f"Reconstruction ({percentage})")
            if kwargs['title_2'] :
                ax2.set_title(kwargs['title_2'])
            else :
                ax2.set_title('Histogram Comparison')
            ax2.set_xlabel('Pixel Intensity')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.set_xlim(xrange[0], xrange[1])
            ax2.set_ylim(yrange[0], yrange[1])

        plt.tight_layout()
        plt.show()
