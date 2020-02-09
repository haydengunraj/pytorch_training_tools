from torch.utils.data import Dataset

TRAIN_SUBSET = 'train'
VAL_SUBSET = 'val'
TEST_SUBSET = 'test'


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
    def __init__(self, dataset, subset):
        self.dataset = dataset
        self.subset = subset

    def __getitem__(self, index):
        return self.dataset[self.dataset.indices[self.subset][index]]

    def __len__(self):
        return len(self.dataset.indices[self.subset])
