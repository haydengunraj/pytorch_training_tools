from torch.utils.data import DataLoader

from ..utils import LOSS_KEY
from ..metrics import get_metrics


def get_trainer(config):
    return Trainer(**config)


class Trainer:
    """Basic trainer class

    Args:
        model (nn.Module, CallableWrapper): The model to be trained.
        dataset (Dataset, DatasetWrapper): The training dataset.
        optimizer (optim.Optimizer): The parameter optimizer.
        loss_func (callable): The loss function.
        writer (SummaryWriter, optional): A SummaryWriter for logging data
            (defaults to None).
        batch_size (int, optional): The training batch size (defaults to 1).
        num_workers (int, optional): The number of workers to use for
            loading data (defaults to 1).
        log_interval: (int, optional): The interval in steps for which
            loss is logged and printed (defaults to 1).
        tag_prefix (str, optional): The prefix appended to logging tags
            (defaults to train/).

    Methods:
        train_epoch(): Run training for a single epoch.
    """
    def __init__(self, model, dataset, optimizer, loss_func, metrics=(), writer=None,
                 batch_size=1, num_workers=1, log_interval=1, tag_prefix='train/'):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.metrics = get_metrics(metrics)
        self.writer = writer
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.log_interval = log_interval
        self.tag_prefix = tag_prefix
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size,
                                     shuffle=True, num_workers=self.num_workers)

    def train_epoch(self, epoch, step, device='cpu'):
        """Train for one epoch"""
        # Store current model state
        is_training = self.model.training
        self.model.train()

        # Train for an epoch
        print('\nStarting epoch {}'.format(epoch))
        for data_batch in self.dataloader:
            self.optimizer.zero_grad()
            data_dict = {key: val.to(device) for key, val in data_batch.items()}

            # Add model outputs to data dict
            data_dict.update(self.model(data_dict))

            # Add loss to data dict
            data_dict.update(self.loss_func(data_dict))

            # Backpropagate loss
            data_dict[LOSS_KEY].backward()
            self.optimizer.step()

            # Log primary loss value
            step += 1
            self._log_and_print(epoch, step, data_dict)

        # Restore initial model state
        self.model.train(is_training)

        return step

    def _log_and_print(self, epoch, step, data_dict):
        log_str = '[epoch: {}, step: {}'.format(epoch, step)
        for metric in self.metrics:
            metric.update(data_dict)
            if not step % self.log_interval:
                if self.writer is not None:
                    metric.log(self.writer, step, self.tag_prefix)
                log_str += ', {}: {:.3f}'.format(metric.name, metric.value)
                metric.reset()
        if not step % self.log_interval:
            print(log_str + ']', flush=True)
