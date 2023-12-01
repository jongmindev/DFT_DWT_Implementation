from abc import ABC, abstractmethod
import numpy as np
import utils
from utils import c_wavelets
from tqdm import tqdm

class Transform(ABC):
    def __init__(self, image_or_path) -> None:
        self.loader = utils.ImageLoader(image_or_path)
        self.image = self.loader.image_grayscaled
        self.transformed_matrix = None
    
    @abstractmethod
    def transform(self) -> np.ndarray:
        pass

    @abstractmethod
    def inverse_transform(self, transformed_matrix: np.ndarray) -> np.ndarray:
        pass

    def compress_matrix(self, retention_ratio, take_abs = True) -> np.ndarray:
        """ 
        compresses the matrix by setting the values below percentage to 0
        * retention_ratio : percentage of the values to keep
        * matrix : matrix to compress (expected to be haar transformed matrix)
        * take_abs : whether compress based on absolute value or remove FROM negative values (i.e. |mask| > threshold or mask > threshold)
        """
        assert self.transformed_matrix is not None, "Do transform first."

        if take_abs:
            threshold = np.percentile(np.abs(self.transformed_matrix), float(retention_ratio))
            mask = np.abs(self.transformed_matrix) > threshold
        else : 
            threshold = np.percentile(self.transformed_matrix, float(retention_ratio))
            mask = self.transformed_matrix > threshold
        matrix_compressed = self.transformed_matrix * mask       
        return matrix_compressed
    
    def compress_image(self, retention_ratio, take_abs = True) -> np.ndarray:
        matrix_compressed = self.compress_matrix(retention_ratio, take_abs)
        reconstructed_image = self.inverse_transform(matrix_compressed)
        return reconstructed_image

    def cwt_compress_matrix(self, retention_ratio, take_abs = True) -> np.ndarray:
        """ 
        compresses the matrix by setting the values below percentage to 0
        * retention_ratio : percentage of the values to keep
        * matrix : matrix to compress (expected to be haar transformed matrix)
        * take_abs : whether compress based on absolute value or remove FROM negative values (i.e. |mask| > threshold or mask > threshold)
        """
        assert self.transformed_matrix is not None, "Do transform first."
        _, _, z_size = self.transformed_matrix.shape
        mask = np.zeros(self.transformed_matrix.shape)
        for scale_idx in range(z_size):
            matrix_scaled = self.transformed_matrix[scale_idx]
            if take_abs:
                threshold = np.percentile(np.abs(matrix_scaled), float(retention_ratio))
                mask = np.abs(matrix_scaled) > threshold
            else : 
                threshold = np.percentile(matrix_scaled, float(retention_ratio))
                mask = matrix_scaled > threshold
        matrix_compressed = self.transformed_matrix * mask       
        return matrix_compressed

    def cwt_compress_image(self, retention_ratio, take_abs = True) -> np.ndarray:
        matrix_compressed = self.cwt_compress_matrix(retention_ratio, take_abs)
        reconstructed_image = self.inverse_transform(matrix_compressed)
        return reconstructed_image

class FFT(Transform):
    def __init__(self, image_or_path) -> None:
        super().__init__(image_or_path)
        self.transformed_matrix = self.transform()
    
    def transform(self) -> np.ndarray:
        return np.fft.fft2(self.image)
    
    def inverse_transform(self, transformed_matrix: np.ndarray) -> np.ndarray:
        return np.fft.ifft2(transformed_matrix).real


class HaarWT(Transform):
    def __init__(self, image_or_path) -> None:
        super().__init__(image_or_path)
        self.image = self.pad()
        self.transformed_matrix = self.transform()
    
    def transform(self) -> np.ndarray:
        x_size, y_size = self.image.shape
        normalized_haar_mat_x = self.normalize(self.get_haar_mat(x_size))
        normalized_haar_mat_y = self.normalize(self.get_haar_mat(y_size))
        transformed_2d = (normalized_haar_mat_x.T) @ self.image @ normalized_haar_mat_y
        return transformed_2d
    
    def inverse_transform(self, transformed_matrix: np.ndarray) -> np.ndarray:
        """ 
        inverse haar transform
        * size : size of the image (transformed haar matrix)
        * img_trans : transformed image (after haar transform)
        * to_unit8 : whether to convert to unit8 (for displaying purposes)
        """
        x_size, y_size = transformed_matrix.shape
        normalized_haar_mat_x = self.normalize(self.get_haar_mat(x_size))
        normalized_haar_mat_y = self.normalize(self.get_haar_mat(y_size))
        transformed_2d = (normalized_haar_mat_x) @ transformed_matrix @ (normalized_haar_mat_y.T)

        h, w = self.loader.image_grayscaled.shape
        transformed_2d = transformed_2d[:h, :w]
        return transformed_2d.astype(np.uint8) # if to_unit8 else transformed_2d

    @staticmethod
    def find_power_of_two(input_number):
        power_of_two = 1
        while power_of_two < input_number:
            power_of_two *= 2
        return power_of_two

    def pad(self):
        # only pads grayscaled image
        # used to pad with zeros (0) so that the image size is a power of 2 #size : (row, col)
        row, col = self.loader.image_grayscaled.shape
        row_ext = self.find_power_of_two(row)
        col_ext = self.find_power_of_two(col)
        row_pad = row_ext - row
        col_pad = col_ext - col
        return np.pad(self.loader.image_grayscaled, ((0,row_pad),(0,col_pad)), 'constant', constant_values=0)

    @staticmethod
    def n_to_size(n: int) -> int:
        return 2 ** n
    
    @staticmethod
    def create_zero_vector(n, i, value):
        """
        creates an (almost) zero vector of size n, with value at index i
        """
        vector = np.zeros(n)
        vector[i] = value
        return vector
    
    def D_n(self, size:int, n:int) -> np.ndarray :
        """
        for normalizing the haar matrix
        * size : size of the vector
        * n : n in the equation (log_2 thing)
        """
        size_log = int(np.log2(size))
        if n == 0 :
            return self.create_zero_vector(size, 0, 2**(-size_log/2)).reshape(1,size)
        else : 
            how_many = int(2**(n-1))
            left_mat = self.D_n(size, n-1)
            right_mat = np.array([self.create_zero_vector(size, x , 2**(-(size_log-(n-1))/2)) for x in range(how_many,how_many+how_many)])
            return np.concatenate((left_mat, right_mat), axis = 0)
    
    def get_haar_mat(self, size: int) -> np.ndarray: #! put in normalize! (with D) 
        # only allow n of power of 2 and n > 0 (check with bitwise and operator)
        assert size & (size - 1) == 0 and size >= 0, "n must be a zero or positive power of 2"
        # get n in the equation (cause 2^n = size)
        n = int(np.log2(size))

        # base case
        if n == 0:
            return np.array([[1]])
        # recursive case, use np.kron to get kronecker product
        else : 
            left_mat = np.kron(self.get_haar_mat(self.n_to_size(n - 1)), np.array([[1], [1]])) #reason for [[1],[-1]] : to make it explicilty into column vector
            right_mat = np.kron(np.eye(self.n_to_size(n - 1)), np.array([[1], [-1]]))
            return np.concatenate((left_mat, right_mat), axis = 1)

    def normalize(self, haar_mat: np.ndarray) -> np.ndarray:
        W_n = haar_mat
        size = W_n.shape[0]
        n = int(np.log2(size))
        H_n = W_n @ self.D_n(size, n)
        return H_n
    

class CWT(Transform):
    def __init__(self, image_or_path) -> None:
        super().__init__(image_or_path)
        self.scales = np.geomspace(1.0,256.0,100)
        self.transformed_matrix, self.wave_norm = self.transform()

    def _get_wavelet_mask(self, wavelet: str, omega_x: np.array, omega_y: np.array, **kwargs):
        assert omega_x.shape == omega_y.shape

        try:
            return c_wavelets[wavelet](omega_x, omega_y, **kwargs)

        except KeyError:
            raise WaveletTransformException('Unknown wavelet: {}'.format(wavelet))

    def _create_frequency_plane(self, image_shape: tuple):
        assert len(image_shape) == 2

        h, w = image_shape
        w_2 = (w - 1) // 2
        h_2 = (h - 1) // 2

        w_pulse = 2 * np.pi / w * np.hstack((np.arange(0, w_2 + 1), np.arange(w_2 - w + 1, 0)))
        h_pulse = 2 * np.pi / h * np.hstack((np.arange(0, h_2 + 1), np.arange(h_2 - h + 1, 0)))

        xx, yy = np.meshgrid(w_pulse, h_pulse, indexing='xy')
        dxx_dyy = abs((xx[0, 1] - xx[0, 0]) * (yy[1, 0] - yy[0, 0]))

        return xx, yy, dxx_dyy

    def transform(self, wavelet='mexh', **wavelet_args):
        assert isinstance(self.image, np.ndarray) and len(self.image.shape) == 2, 'x should be 2D numpy array'

        x_image = np.fft.fft2(self.image)
        xx, yy, dxx_dyy = self._create_frequency_plane(x_image.shape)

        cwt = []
        wav_norm = []

        for scale_val in tqdm(self.scales):
            mask = scale_val * self._get_wavelet_mask(wavelet, scale_val * xx, scale_val * yy, **wavelet_args)

            cwt.append(np.fft.ifft2(x_image * mask))
            wav_norm.append((np.sum(abs(mask)**2)*dxx_dyy)**(0.5 / (2 * np.pi)))

        cwt = np.stack(cwt, axis=2)
        wav_norm = np.array(wav_norm)

        return cwt, wav_norm

    def inverse_transform(self, compressed_matrix: np.ndarray) -> np.ndarray:
        # C = 1.0 / (self.scales * self.wave_norm)
        # reconstruction = (C * np.real(compressed_matrix)).sum(axis=-1)
        # reconstruction = 1.0 - (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
        # return reconstruction
        pass
 