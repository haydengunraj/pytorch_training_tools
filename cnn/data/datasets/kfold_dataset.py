import os
import sys
import numpy as np
from torchvision.datasets.folder import make_dataset

from .split_dataset import SplitDataset, TRAIN_SUBSET, VAL_SUBSET


class KFoldDatasetFolder(SplitDataset):
    """A generic data loader where the samples are arranged in this way:

        root/fold_prefix0/class_x/xxx.ext
        root/fold_prefix0/class_y/xxy.ext
        root/fold_prefix0/class_z/xxz.ext

        root/fold_prefix1/class_x/123.ext
        root/fold_prefix1/class_y/nsdf3.ext
        root/fold_prefix1/class_z/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        k (int): The number of folds
        holdout (int, optional): The index of the holdout fold.
            Defaults to no holdout.
        fold_prefix (str, optional): The prefix for fold directories
            Defaults to 'fold', meaning it will search for directories
            of the form 'fold0', 'fold1', etc.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.

    Methods:
        get_train_dataset(): Returns the train dataset
        get_val_dataset(): Returns the val dataset

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """
    def __init__(self, root, loader, folds, holdout=None, fold_prefix='fold', extensions=None,
                 transform=None, target_transform=None, is_valid_file=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        classes, class_to_idx = self._find_classes(os.path.join(self.root, fold_prefix + str(0)))
        samples = []
        fold_indices = []
        for fold in range(folds):
            fold_root = os.path.join(self.root, fold_prefix + str(fold))
            fold_samples = make_dataset(fold_root, class_to_idx, extensions, is_valid_file)
            samples.extend(fold_samples)
            fold_indices.extend([fold]*len(fold_samples))
        if len(samples) == 0:
            raise RuntimeError('Found 0 files in subfolders of: ' + self.root + '\n'
                               'Supported extensions are: ' + ','.join(extensions))

        self.loader = loader
        self.extensions = extensions

        self.folds = folds
        self.holdout = holdout
        self.fold_indices = np.asarray(fold_indices)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.set_holdout(self.holdout)

    def set_holdout(self, holdout=None):
        """Specifies the holdout set index"""
        if holdout is None:
            self.holdout = holdout
            self.indices[TRAIN_SUBSET] = list(range(len(self.samples)))
            self.indices[VAL_SUBSET] = []
        else:
            if holdout < 0 or holdout >= self.folds:
                raise ValueError('Holdout set must be an integer in the range [0, {})'.format(self.folds))
            train_mask = self.fold_indices != holdout
            self.holdout = holdout
            self.indices[TRAIN_SUBSET] = list(np.where(train_mask)[0])
            self.indices[VAL_SUBSET] = list(np.where(~train_mask)[0])

    @staticmethod
    def _find_classes(directory):
        """Finds the class folders in a dataset"""
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(directory) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return len(self.samples)
