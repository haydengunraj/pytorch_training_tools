from torch.utils.data import Dataset
import numpy as np
from math import ceil

TRAIN_SUBSET = 'train'
VAL_SUBSET = 'val'
TEST_SUBSET = 'test'


def create(_):
    raise ValueError('SplitDataset may not be used directly')


class SplitDataset:
    """Mixin class for datasets which divide data into train/val/test splits

    Methods:
        get_train_dataset(): Returns the train dataset
        get_val_dataset(): Returns the val dataset
        get_test_dataset(): Returns the test dataset
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.indices = {TRAIN_SUBSET: [], VAL_SUBSET: [], TEST_SUBSET: []}
        self.samples = getattr(self, 'samples', [])
        self.splittable = True

    def split(self, train_fraction=0.8, val_fraction=0.2, test_fraction=0, seed=1):
        """Splits data into train/val/test subsets"""
        if self.is_initialized():
            return
        self.ensure_fraction_sum(train_fraction, val_fraction, test_fraction)
        np.random.seed(seed)
        self.samples = sorted(self.samples)
        np.random.shuffle(self.samples)
        train_idx = ceil(train_fraction*(len(self.samples)))
        val_idx = train_idx + ceil(val_fraction*(len(self.samples)))
        test_idx = val_idx + ceil(test_fraction*(len(self.samples)))
        indices = list(range(len(self.samples)))
        self.indices[TRAIN_SUBSET] = indices[:train_idx]
        self.indices[VAL_SUBSET] = indices[train_idx:val_idx]
        self.indices[TEST_SUBSET] = indices[val_idx:test_idx]

    def is_initialized(self):
        return sum(len(subset) for subset in self.indices.values()) > 0

    @staticmethod
    def ensure_fraction_sum(*subset_fractions):
        if sum(subset_fractions) > 1:
            raise ValueError('Sum of subset fractions must be less than or equal to 1')

    def get_train_dataset(self):
        if not len(self.indices[TRAIN_SUBSET]):
            raise ValueError('Train subset not initialized')
        return _SplitSubset(self, TRAIN_SUBSET)

    def get_val_dataset(self):
        if not len(self.indices[VAL_SUBSET]):
            raise ValueError('Validation subset not initialized')
        return _SplitSubset(self, VAL_SUBSET)

    def get_test_dataset(self):
        if not len(self.indices[TEST_SUBSET]):
            raise ValueError('Test subset not initialized')
        return _SplitSubset(self, TEST_SUBSET)


class _SplitSubset(Dataset):
    """A subset of a split dataset"""
    def __init__(self, dataset, subset):
        self.dataset = dataset
        self.subset = subset

    def __getitem__(self, index):
        return self.dataset[self.dataset.indices[self.subset][index]]

    def __len__(self):
        return len(self.dataset.indices[self.subset])
