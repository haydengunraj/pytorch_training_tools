from torch.utils.data import Dataset
from ...utils import map_outputs


class DatasetWrapper:
    def __init__(self, dataset, output_keys):
        self.dataset = dataset
        self.output_keys = output_keys

    def __getattr__(self, attr):
        return getattr(self.__dict__['dataset'], attr)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __getitem__(self, index):
        return map_outputs(self.output_keys, self.dataset[index])

    def __len__(self):
        return len(self.dataset)
