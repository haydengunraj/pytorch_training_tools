import torch
from .metric import LogMetric

MODE = None


def create(config):
    return ImageLogMetric(**config)


class ImageLogMetric(LogMetric):
    """Stores images for logging purposes"""
    def __init__(self, name, image_key='images', max_images=10, mean=None, std=None):
        super().__init__(name)
        self.image_key = image_key
        self.max_images = max_images
        self.mean = mean
        self.std = std
        self.images = []

    def update(self, data_dict):
        if len(self.images) < self.max_images:
            image = self.denormalize(data_dict[self.image_key][0])
            self.images.append(image)

    def reset(self):
        self.images = []

    def log(self, writer, step, tag_prefix='val/'):
        for i, image in enumerate(self.images):
            writer.add_image('{}{}_{}'.format(tag_prefix, self.name, i), image, step)

    def denormalize(self, image):
        if self.std is not None:
            std = torch.as_tensor(self.std, dtype=image.dtype, device=image.device)
            image = image*std[:, None, None]
        if self.mean is not None:
            mean = torch.as_tensor(self.mean, dtype=image.dtype, device=image.device)
            image = image + mean[:, None, None]
        return image
