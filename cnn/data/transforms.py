import random
import numpy as np
from PIL import Image


class Unsqueeze:
    """Add a new dimension"""
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, tensor):
        return tensor.unsqueeze(self.dim)

    def __repr__(self):
        return self.__class__.__name__ + '({})'.format(self.dim)


class RandomPermute:
    """Randomly permutes the given axes"""
    def __init__(self, p=0.5, dims=(1, 2)):
        self.p = p
        self.dims = dims

    def __call__(self, tensor):
        if self.p < random.random():
            return tensor
        dims = list(range(tensor.ndim))
        perm_dims = list(self.dims)
        random.shuffle(perm_dims)
        for orig, new in zip(self.dims, perm_dims):
            dims[orig] = new
        return tensor.permute(*dims)


class ResizeWithPad:
    """Resizes to square images while maintaining aspect ratio using padding"""
    def __init__(self, size, pad_value=0):
        self.pad_value = pad_value
        if isinstance(size, int):
            self.width = size
            self.height = size
        elif isinstance(size, tuple) and len(size) == 2:
            self.width = size[0]
            self.height = size[1]
        else:
            raise ValueError('size must be an int or 2-tuple')

    def __call__(self, pil_image):
        w, h = pil_image.size
        scale = min(self.width/w, self.height/h)
        new_size = (int(np.floor(w*scale)), int(np.floor(h*scale)))
        resized_arr = np.asarray(pil_image.resize(new_size))
        new_array = np.ones((self.height, self.width, len(pil_image.getbands())), dtype=np.uint8)*int(self.pad_value)
        new_array[:new_size[1], :new_size[0], ...] = resized_arr
        return Image.fromarray(new_array)
