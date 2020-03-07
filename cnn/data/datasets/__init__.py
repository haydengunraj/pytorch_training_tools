from PIL import ImageFile
import torchvision.datasets as datasets

from .wrapper import DatasetWrapper
from .. import get_transform
from ...utils import import_submodule


# Allow truncated image files to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataset(config):
    dataset_type = config.pop('type')
    output_keys = config.pop('output_keys')
    if config.get('transform') is not None:
        config['transform'] = get_transform(config['transform'])
    if config.get('target_transform') is not None:
        config['target_transform'] = get_transform(config['target_transform'])
    module = import_submodule(__name__, dataset_type)
    if module is not None:
        dataset = module.get_dataset(config)
    else:
        dataset = getattr(datasets, dataset_type)
        if dataset is None:
            raise ValueError('Unrecognized dataset type: ' + dataset_type)
        dataset = dataset(**config)
    return DatasetWrapper(dataset, output_keys)
