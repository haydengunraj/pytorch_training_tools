import os
import yaml
import json
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from tensorboardX import SummaryWriter

from cnn.models import get_model
from cnn.data.datasets import get_dataset
from cnn.losses import get_loss as _get_loss
from cnn.trainers import get_trainer as _get_trainer
from cnn.evaluators import get_evaluator as _get_evaluator
from cnn.saver import CheckpointSaver

TORCH_HOME = 'weights'
EXPERIMENT_ROOT = 'experiments'


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
    if getattr(train_dataset, 'splittable', False):
        val_dataset = train_dataset.get_val_dataset()
        train_dataset = train_dataset.get_train_dataset()
    else:
        val_dataset = get_dataset(config['val_dataset']) if config.get('val_dataset') is not None else None

    # Initialize loss function
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


def _get_component(module, config):
    comp_type = config.pop('type')
    component = getattr(module, comp_type, None)
    if component is None:
        raise ValueError('Unrecognized component type: {}'.format(comp_type))
    return component(**config)


def get_loss(config, device):
    loss_weight = config.get('weight')
    if loss_weight is not None:
        config['weight'] = torch.FloatTensor(loss_weight).to(device)
    return _get_loss(config)


def get_optimizer(config, model):
    config['params'] = model.parameters()
    optimizer = _get_component(optim, config)
    return optimizer


def get_scheduler(config, optimizer):
    config['optimizer'] = optimizer
    scheduler = _get_component(lr_scheduler, config)
    return scheduler


def get_writer(config, experiment_dir, start_step):
    config['logdir'] = os.path.join(experiment_dir, 'events')
    config['purge_step'] = start_step if start_step > 0 else None
    config.pop('type', None)
    writer = SummaryWriter(**config)
    return writer


def get_trainer(config, model, dataset, optimizer, loss_func, writer):
    config['model'] = model
    config['dataset'] = dataset
    config['optimizer'] = optimizer
    config['loss_func'] = loss_func
    config['writer'] = writer
    trainer = _get_trainer(config)
    return trainer


def get_evaluator(config, model, dataset, loss_func, writer):
    if dataset is None:
        return None
    config['model'] = model
    config['dataset'] = dataset
    config['loss_func'] = loss_func
    config['writer'] = writer
    evaluator = _get_evaluator(config)
    return evaluator


def get_saver(config, experiment_dir, model, optimizer, scheduler):
    config['checkpoint_dir'] = os.path.join(experiment_dir, 'checkpoints')
    config['model'] = model
    config['optimizer'] = optimizer
    config['scheduler'] = scheduler
    config.pop('type', None)
    saver = CheckpointSaver(**config)
    return saver
