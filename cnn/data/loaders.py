import numpy as np
from torchvision.datasets.folder import pil_loader

# Allow truncated image files to be loaded
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

"""Wrapper module for loading functions"""
numpy = np.load
pil = pil_loader
