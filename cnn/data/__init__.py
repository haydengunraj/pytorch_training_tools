import torchvision.transforms as tv_transforms

from . import transforms


def _get_transform(config):
    transform_type = config.pop('type')
    transform = getattr(transforms, transform_type, None)
    if transform is None:
        transform = getattr(tv_transforms, transform_type, None)
    if transform is None:
        raise ValueError('Unrecognized transform type: ' + transform_type)
    return transform(**config)


def get_transform(transform_list):
    transform = []
    for config in transform_list:
        transform.append(_get_transform(config))
    return tv_transforms.Compose(transform)
