from torch.utils.data import Dataset
from ...utils import map_outputs


class DatasetWrapper(Dataset):
    def __init__(self, dataset, output_keys):
        self.dataset = dataset
        self.output_keys = output_keys

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def __getitem__(self, index):
        return map_outputs(self.output_keys, self.dataset[index])

    def __len__(self):
        return len(self.dataset)
