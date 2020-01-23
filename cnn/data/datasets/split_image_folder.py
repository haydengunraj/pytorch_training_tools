import numpy as np
from torchvision.datasets.folder import ImageFolder

from .split_dataset import SplitDataset, TRAIN_SUBSET, VAL_SUBSET


class SplitImageFolder(ImageFolder, SplitDataset):
    """A wrapper class which splits an ImageFolder dataset into train/test sets

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

    Methods:
        get_train_dataset(): Returns the train dataset
        get_val_dataset(): Returns the val dataset

    """
    def __init__(self, root, train_fraction=0.8, transform=None, target_transform=None, seed=None):
        super().__init__(root, transform=transform, target_transform=target_transform)

        if train_fraction < 0 or train_fraction > 1:
            raise ValueError('train_fraction must be in the range [0, 1]')
        self.train_fraction = train_fraction
        self._split(seed)

    def _split(self, seed):
        """Splits the data according to self.train_fraction"""
        if len(self.indices[TRAIN_SUBSET]) or len(self.indices[VAL_SUBSET]):
            return
        if seed is not None:
            np.random.seed(seed)
        targets = np.asarray(self.targets)
        for cls in self.classes:
            idx = self.class_to_idx[cls]
            in_class = np.where(targets == idx)[0]
            n_in_class = len(in_class)
            split_idx = int(self.train_fraction*n_in_class)
            np.random.shuffle(in_class)
            self.indices[TRAIN_SUBSET].extend(list(in_class[:split_idx]))
            self.indices[VAL_SUBSET].extend(list(in_class[split_idx:]))
