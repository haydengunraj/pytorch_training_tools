from PIL import ImageFile
import torchvision.datasets as datasets

from .. import get_transform
from ...utils import get_component, DatasetWrapper

# Allow truncated image files to be loaded
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_dataset(config):
    dataset_type = config.pop('type')
    output_keys = config.pop('output_keys')
    if config.get('transform') is not None:
        config['transform'] = get_transform(config['transform'])
    if config.get('target_transform') is not None:
        config['target_transform'] = get_transform(config['target_transform'])
    dataset = get_component(__name__, dataset_type, config, (datasets,))
    return DatasetWrapper(dataset, output_keys)
