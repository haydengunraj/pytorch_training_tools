import os
import sys
import numpy as np
from torchvision.datasets.folder import VisionDataset, make_dataset
from torch.utils.data import Dataset


class KFoldDatasetFolder(VisionDataset):
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
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                "Supported extensions are: " + ",".join(extensions)))

        self.loader = loader
        self.extensions = extensions

        self.folds = folds
        self.holdout = holdout
        self.fold_indices = np.asarray(fold_indices)

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.all_samples = samples
        self.all_targets = [s[1] for s in samples]
        self.samples = []
        self.targets = []

        self.holdout_dataset = None
        self.set_holdout(self.holdout)

    def set_holdout(self, holdout=None):
        """Specifies the holdout set index"""
        if holdout is None:
            self.samples = self.all_samples
            self.targets = self.all_targets
            self.holdout = holdout
            self.holdout_dataset = None
        else:
            if holdout < 0 or holdout >= self.folds:
                raise ValueError('Holdout set must be an integer in the range [0, {})'.format(self.folds))
            not_holdout = np.where(self.fold_indices != holdout)[0]
            self.samples = [self.all_samples[i] for i in not_holdout]
            self.targets = [self.all_targets[i] for i in not_holdout]
            self.holdout = holdout
            self.holdout_dataset = HoldoutDataset(self, list(np.where(self.fold_indices == self.holdout)[0]))
            self.holdout_dataset.classes = self.classes

    def load_and_transform(self, path, target):
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    @staticmethod
    def _find_classes(dir):
        """Finds the class folders in a dataset"""
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample, target = self.load_and_transform(path, target)
        return sample, target

    def __len__(self):
        return len(self.samples)


class HoldoutDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        path, target = self.dataset.all_samples[idx]
        sample, target = self.dataset.load_and_transform(path, target)
        return sample, target

    def __len__(self):
        return len(self.indices)
