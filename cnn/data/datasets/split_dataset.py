from torch.utils.data import Dataset
from torchvision.datasets.folder import VisionDataset

TRAIN_SUBSET = 'train'
VAL_SUBSET = 'val'


class SplitDataset(VisionDataset):
    """Base class for datasets which divide data into train and val splits

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
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.indices = {TRAIN_SUBSET: [], VAL_SUBSET: []}

    def get_train_dataset(self):
        if not len(self.indices[TRAIN_SUBSET]):
            return None
        return _SplitSubset(self, TRAIN_SUBSET)

    def get_val_dataset(self):
        if not len(self.indices[VAL_SUBSET]):
            return None
        return _SplitSubset(self, VAL_SUBSET)

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class _SplitSubset(Dataset):
    def __init__(self, dataset, subset):
        self.dataset = dataset
        self.subset = subset
        self.__dict__.update(self.dataset.__dict__)

    def __getitem__(self, index):
        return self.dataset[self.dataset.indices[self.subset][index]]

    def __len__(self):
        return len(self.dataset.indices[self.subset])
