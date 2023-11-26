import numpy as np
import utils
from typing import Union
#from skimage.transform import resize

class Load_Image:
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

    #not actually used (would be cheating since skimage was used... but just for testing purposes (scale down, and see O graph))        
    #def resize(self, size: tuple):
    #    #only resizes grayscaled image
    #    self.image_grayscaled = resize(self.image_grayscaled, size)
    
    def pad(self,size: tuple):
        #only pads grayscaled image
        #used to pad with zeros (0) so that the image size is a power of 2 #size : (row, col)
        row, col = self.image_grayscaled.shape
        assert row <= size[0] and col <= size[1], "Invalid size : size must be larger than the original image"
        row_pad = size[0] - row
        col_pad = size[1] - col
        self.image_grayscaled = np.pad(self.image_grayscaled, ((0,row_pad),(0,col_pad)), 'constant', constant_values=0)

class HaarTransform(Load_Image):
    def __init__(self, image_or_path: Union[np.ndarray, str]) -> None:
        super().__init__(image_or_path)

    def n_to_size(self, n: int) -> int:
        return 2 ** n

    def create_zero_vector(self,n, i, value):
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

    def get_haar_mat(self,size: int) -> np.ndarray: #! put in normalize! (with D) 
        # only allow n of power of 2 and n > 0 (check with bitwise and operator)
        assert size & (size - 1) == 0 and size >= 0, "n must be a zero or positive power of 2"
        #get n in the equation (cause 2^n = size)
        n = int(np.log2(size))

        # base case
        if n == 0:
            return np.array([[1]])
        else : 
            # recursive case, use np.kron to get kronecker product
            left_mat = np.kron(self.get_haar_mat(self.n_to_size(n - 1)), np.array([[1], [1]])) #reason for [[1],[-1]] : to make it explicilty into column vector
            right_mat = np.kron(np.eye(self.n_to_size(n - 1)), np.array([[1], [-1]]))
            return np.concatenate((left_mat, right_mat), axis = 1) 
    
    def normalize(self, haar_mat: np.ndarray) -> np.ndarray:
        W_n = haar_mat
        size = W_n.shape[0]
        n = int(np.log2(size))
        H_n = W_n@self.D_n(size, n)
        return H_n
    
    def haar_transform_2d(self, size: tuple, img : np.ndarray) -> np.ndarray:
        x_size, y_size = size
        normalized_haar_mat_x = self.normalize(self.get_haar_mat(x_size))
        normalized_haar_mat_y = self.normalize(self.get_haar_mat(y_size))
        transformed_2d = (normalized_haar_mat_x.T)@img@normalized_haar_mat_y
        return transformed_2d


print("hi")
haar = HaarTransform("./Boneyard_IMG_5341.jpg")
power_2_shape = (4096, 8192)
haar.pad(power_2_shape)
padded_image = haar.image_grayscaled
aa = haar.haar_transform_2d(size = power_2_shape, img = padded_image)

print(aa.shape)