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


def initialize_experiment_kfold(experiment, holdout):
    """ Initialize an experiment using a kfold dataset """
    # Get experiment directory and config
    config, experiment_dir = load_config(experiment)

    # Check dataset type
    if config['train_dataset']['type'] != 'KFoldImageFolder':
        raise ValueError('KFold experiements must use a KFoldImageFolder as the train dataset')

    # Set holdout and check number of folds
    config['train_dataset']['holdout'] = holdout
    n_folds = config['train_dataset']['folds']

    # Change logging and save directories
    if 'writer' in config:
        config['writer']['logdir'] = os.path.join(config['writer']['logdir'], 'holdout{}'.format(holdout))
    if 'saver' in config:
        config['saver']['checkpoint_dir'] = os.path.join(config['saver']['checkpoint_dir'], 'holdout{}'.format(holdout))

    # Intitialize modules
    return _initialize(experiment_dir, config) + (n_folds,)


def initialize_experiment(experiment, resume=False, reset_optimizer=False, reset_scheduler=False):
    # Get experiment directory and config
    config, experiment_dir = load_config(experiment)

    # Intitialize modules
    return _initialize(experiment_dir, config, resume=resume, reset_optimizer=reset_optimizer,
                       reset_scheduler=reset_scheduler)


def _initialize(experiment_dir, config, resume=False, reset_optimizer=False, reset_scheduler=False):
    """ Initializes training modules """
    # Get device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print('Using device {}'.format(device))

    # Set torch home dir
    os.environ['TORCH_HOME'] = TORCH_HOME

    # Epoch and step counters
    start_epoch = 0
    start_step = 0

    # Initialize model
    model = get_model(config['model'])
    model.to(device)

    # Initialize train dataset
    train_dataset = get_dataset(config['train_dataset'])

    # Check for SplitDataset and initialize val dataset
    if isinstance(train_dataset, data.SplitDataset):
        val_dataset = train_dataset.get_val_dataset()
        train_dataset = train_dataset.get_train_dataset()
    else:
        val_dataset = get_dataset(config['val_dataset']) if config.get('val_dataset') is not None else None

    # Initialize loss function (if necessary)
    loss_func = get_loss(config['loss'], device)

    # Initialize optimizer
    optimizer = get_optimizer(config['optimizer'], model)

    # Initialize scheduler
    scheduler = get_scheduler(config['scheduler'], optimizer) if config.get('scheduler') is not None else None

    # Resume from existing checkpoint if desired
    if resume:
        start_epoch, start_step = load_checkpoint(experiment_dir, model, optimizer,
                                                  scheduler, reset_optimizer, reset_scheduler)

    # Initialize event writer
    writer = get_writer(config['writer'], experiment_dir, start_step) if config.get('writer') is not None else None

    # Initialize trainer
    trainer = get_trainer(config['trainer'], model, train_dataset, optimizer, loss_func, writer)

    # Initialize evaluator
    evaluator = get_evaluator(config['evaluator'], model, val_dataset, loss_func, writer) \
        if config.get('evaluator') is not None else None

    # Initialize checkpoint saver
    checkpoint_saver = get_saver(config['saver'], experiment_dir, model, optimizer, scheduler) \
        if config.get('saver') is not None else None

    return (model, train_dataset, val_dataset, trainer, evaluator, scheduler, loss_func,
            optimizer, checkpoint_saver, writer, device, start_epoch, start_step)


def load_config(experiment):
    experiment_dir = os.path.join(EXPERIMENT_ROOT, experiment)
    config_file = os.path.join(experiment_dir, 'config.yml')
    with open(config_file) as f:
        config = yaml.safe_load(f)
    return config, experiment_dir


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def load_checkpoint(experiment_dir, model, optimizer, scheduler, reset_optimizer=False, reset_scheduler=False):
    checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
    checkpoint_file = os.path.join(checkpoint_dir, 'checkpoint.json')
    if os.path.isfile(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            ckpt = json.load(f)
        state_dict = torch.load(os.path.join(checkpoint_dir, ckpt['checkpoint']))
        model.load_state_dict(state_dict['model'])
        if not reset_optimizer:
            optimizer.load_state_dict(state_dict['optimizer'])
        if not reset_scheduler and state_dict['scheduler'] is not None:
            scheduler.load_state_dict(state_dict['scheduler'])
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
    tforms = config.get('transform')
    if tforms is not None:
        config['transform'] = transforms.Compose(get_transforms(tforms))
    if config['type'] == 'kfold_dataset':
        config['loader'] = get_component('loader', config['loader'])
        config['extensions'] = tuple(config['extensions'])
    dataset = get_component('dataset', config)
    return dataset


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
