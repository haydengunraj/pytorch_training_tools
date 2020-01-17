import os
import yaml
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms, models
from torchvision.datasets import folder
import tensorboardX

from . import models as cnn_models
from . import data
from .validation import eval
from .training import trainers, losses, saver

TORCH_HOME = 'weights'
EXPERIMENT_ROOT = 'experiments'

MODULES = {
    'model': [cnn_models, models],
    'dataset': [data, folder],
    'loader': [data.loaders],
    'transform': [data.transforms, transforms],
    'loss': [losses, nn],
    'optimizer': [optim],
    'scheduler': [lr_scheduler],
    'trainer': [trainers],
    'evaluator': [eval],
    'saver': [saver],
    'writer': [tensorboardX]
}

# TODO: update to match new style
# def initialize_experiment_kfold(config, holdout):
#     config['train_dataset']['holdout'] = holdout
#     if 'writer' in config:
#         config['writer']['logdir'] = os.path.join(config['writer']['logdir'], 'holdout{}'.format(holdout))
#     if 'saver' in config:
#         config['saver']['checkpoint_dir'] = os.path.join(config['saver']['checkpoint_dir'], 'holdout{}'.format(holdout))
#     return initialize_experiment(config)


def initialize_experiment(experiment, resume=False, reset_optimizer=False, reset_scheduler=False):
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Using device {}'.format(device))

    # Set torch home dir
    os.environ['TORCH_HOME'] = TORCH_HOME

    # Set experiment directory
    experiment_dir = os.path.join(EXPERIMENT_ROOT, experiment)

    # Load experiment configuration
    conf_path = os.path.join(experiment_dir, 'config.yml')
    config = load_config(conf_path)
    start_epoch = 0
    start_step = 0

    # Initialize model
    model = get_model(config['model'])
    model.to(device)

    # Initialize train dataset
    train_dataset = get_dataset(config['train_dataset'])

    # Initialize val dataset
    val_dataset = get_val_dataset(config, train_dataset)

    # Initialize loss function (if necessary)
    loss_func = get_loss(config['loss'], device)

    # Initialize optimizer
    optimizer = get_optimizer(config['optimizer'], model)

    # Initialize scheduler
    scheduler = get_scheduler(config['scheduler'], optimizer) if 'scheduler' in config else None

    # Resume from existing checkpoint if desired
    if resume:
        start_epoch, start_step = load_checkpoint(experiment_dir, model, optimizer,
                                                  scheduler, reset_optimizer, reset_scheduler)

    # Initialize event writer
    writer = get_writer(config['writer'], experiment_dir, start_step) if 'writer' in config else None

    # Initialize trainer
    trainer = get_trainer(config['trainer'], model, train_dataset, optimizer, loss_func, writer)

    # Initialize evaluator
    evaluator = get_evaluator(
        config['evaluator'], model, val_dataset, loss_func, writer) if 'evaluator' in config else None

    # Initialize checkpoint saver
    saver = get_saver(config['saver'], experiment_dir, model, optimizer, scheduler) if 'saver' in config else None

    return (model, train_dataset, val_dataset, trainer, evaluator, scheduler, loss_func,
            optimizer, saver, writer, device, start_epoch, start_step)


def load_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def load_checkpoint(experiment_dir, model, optimizer, scheduler, reset_optimizer=False, reset_scheduler=False):
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.json')
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            ckpt = json.load(f)
        _load_state_dict(model, checkpoint_dir, ckpt.get('model_state'))
        if not reset_optimizer:
            _load_state_dict(optimizer, checkpoint_dir, ckpt.get('optimizer_state'))
        if not reset_scheduler:
            _load_state_dict(scheduler, checkpoint_dir, ckpt.get('scheduler_state'))
        start_epoch = int(ckpt['epoch'])
        start_step = int(ckpt['step'])
        print('Resuming training from checkpoint at epoch {}, step {}'.format(start_epoch, start_step))
    else:
        start_epoch = 0
        start_step = 0
        print('No checkpoint.json file found, training from scratch')
    return start_epoch, start_step


def get_component(name, config):
    type_def = config.pop('type')
    component = None
    for module in MODULES[name]:
        component = getattr(module, type_def, None)
        if component is not None:
            component = component(**config)
            break
    if component is None:
        raise ValueError('Unrecognized type {} for component {}'.format(type_def, name))
    return component


def get_model(config):
    model_type = config['type']
    model = get_component('model', config)
    if model is None:
        model = _torch_model_fetcher(model_type, config)
    return model


def get_dataset(config):
    config['transform'] = transforms.Compose(get_transforms(config['transform']))
    if config['type'] == 'kfold_dataset':
        config['loader'] = get_component('loader', config['loader'])
        config['extensions'] = tuple(config['extensions'])
    dataset = get_component('dataset', config)
    return dataset


def get_val_dataset(config, train_dataset):
    if 'val_dataset' in config:
        return get_dataset(config['val_dataset'])
    if isinstance(train_dataset, data.KFoldDatasetFolder):
        return train_dataset.holdout_dataset
    return None


def get_transforms(config):
    transform = []
    for conf in config:
        transform.append(get_component('transform', conf))
    return transform


def get_loss(config, device):
    weight = config.get('weight')
    if weight is not None:
        config['weight'] = torch.FloatTensor(weight).to(device)
    loss_func = get_component('loss', config)
    return loss_func


def get_optimizer(config, model):
    config['params'] = model.parameters()
    optimizer = get_component('optimizer', config)
    return optimizer


def get_scheduler(config, optimizer):
    config['optimizer'] = optimizer
    scheduler = get_component('scheduler', config)
    return scheduler


def get_writer(config, experiment_dir, start_step):
    config['logdir'] = os.path.join(experiment_dir, 'events')
    config['purge_step'] = start_step if start_step > 0 else None
    writer = get_component('writer', config)
    return writer


def get_trainer(config, model, dataset, optimizer, loss_func, writer):
    config['model'] = model
    config['dataset'] = dataset
    config['optimizer'] = optimizer
    config['loss_func'] = loss_func
    config['writer'] = writer
    trainer = get_component('trainer', config)
    return trainer


def get_evaluator(config, model, dataset, loss_func, writer):
    config['model'] = model
    config['dataset'] = dataset
    config['writer'] = writer
    evaluator = get_component('evaluator', config)
    if 'loss' in config['metrics']:
        evaluator.metrics['loss'] = loss_func
    return evaluator


def get_saver(config, experiment_dir, model, optimizer, scheduler):
    config['checkpoint_dir'] = os.path.join(experiment_dir, 'checkpoints')
    config['model'] = model
    config['optimizer'] = optimizer
    config['scheduler'] = scheduler
    saver = get_component('saver', config)
    return saver


def _torch_model_fetcher(model_type, config):
    weights_path = config.pop('weights_path', None)
    model = getattr(models, model_type)(**config)
    if weights_path is not None:
        model.load_state_dict(torch.load(weights_path))
    return model


def _load_state_dict(component, checkpoint_dir, state_dict_file):
    if component is None or state_dict_file is None:
        return
    component.load_state_dict(torch.load(os.path.join(checkpoint_dir, state_dict_file)))