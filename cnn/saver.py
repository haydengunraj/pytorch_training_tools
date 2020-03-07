import os
import json
import torch

from cnn.metrics import get_mode


class CheckpointSaver:
    """Model checkpoint manager

    Args:
        checkpoint_dir (str): The directory to save checkpoints to.
        model (nn.Module): The model to save.
        optimizer (optim.Optimizer): The optimizer to save.
        scheduler (optim.lr_scheduler._LRScheduler): The LR scheduler to save.
        save_interval (int, optional): The interval in epochs for which
            checkpoints are saved (defaults to 1).
        last_epoch (int, optional): The final epoch number. A checkpoint will
            be saved for this epoch regardless of other settings (defaults to None).
        prefix (str, optional): The prefix for checkpoint files (defaults to
            checkpoint_step_).
        save_best_only (bool, optional): A flag to only save the best checkpoint
            based on an evaluation metric (defaults to True).
        metric (str, optional): The name of the metric to use for selecting the
            best checkpoint (defaults to loss).

    Methods:
        update_checkpoint(): Evaluate the current checkpoint and save it if required.
        save_state(): Save a checkpoint in the current state.
    """
    def __init__(self, checkpoint_dir, model, optimizer, scheduler, save_interval=1, last_epoch=None,
                 prefix='checkpoint_step_', save_best_only=True, metric='loss'):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.json')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_interval = save_interval
        self.last_epoch = last_epoch
        self.prefix = prefix
        self.save_best_only = save_best_only
        self.save_path = None
        self.checkpoint_dict = None
        self.metric = metric
        self.last_metric_val = None
        self.mode = get_mode(metric)

    def update_checkpoint(self, epoch, step, metric_vals=None):
        if epoch == self.last_epoch:
            self.save_state(epoch, step)
            print('Saved checkpoint at epoch {}, step {}'.format(epoch, step))
        elif not epoch % self.save_interval:
            if self.save_best_only:
                if metric_vals is None or self.metric not in metric_vals:
                    print('Metric {} not found in metric vals, defaulting to always saving'.format(self.metric))
                    self.save_best_only = False
                    self.update_checkpoint(epoch, step, metric_vals)
                elif self.is_improved(metric_vals[self.metric]):
                    path = self.save_state(epoch, step)

                    if self.last_metric_val is None:
                        save_str = '{} improved from {} to {:.3f}, saving checkpoint'
                    else:
                        save_str = '{} improved from {:.3f} to {:.3f}, saving checkpoint'
                    print(save_str.format(self.metric.capitalize(), self.last_metric_val, metric_vals[self.metric]))
                    self.last_metric_val = metric_vals[self.metric]

                    if self.save_path is not None:
                        self._remove_previous_checkpoint()
                    self.save_path = path
                else:
                    print('{} worsened from {:.3f} to {:.3f}, not saving checkpoint'.format(
                        self.metric.capitalize(), self.last_metric_val, metric_vals[self.metric]))
            else:
                self.save_state(epoch, step)
                print('Saved checkpoint at epoch {}, step {}'.format(epoch, step))

    def is_improved(self, metric_val):
        if metric_val is None:
            raise ValueError('No metric value given')
        if (self.last_metric_val is None
                or self.mode == 'minimize' and metric_val < self.last_metric_val
                or self.mode == 'maximize' and metric_val > self.last_metric_val):
            return True
        return False

    def set_checkpoint_dir(self, checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.checkpoint_dir = checkpoint_dir

    def save_state(self, epoch, step):
        path, filename = self._make_save_path(step)
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None
        }
        torch.save(state_dict, path)
        self._update_checkpoint_file(filename, epoch, step)
        return path

    def _remove_previous_checkpoint(self):
        try:
            os.unlink(self.save_path)
        except FileNotFoundError:
            print('Could not delete checkpoint file at {}'.format(self.save_path))
            pass

    def _update_checkpoint_file(self, filename, epoch, step):
        self.checkpoint_dict = {
            'checkpoint': filename,
            'epoch': str(epoch),
            'step': str(step)
        }
        self._write_checkpoint_file()

    def _write_checkpoint_file(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint_dict, f)

    def _make_save_path(self, step):
        postfix = '{}.pth'.format(step)
        filename = self.prefix + postfix
        path = os.path.join(self.checkpoint_dir, filename)
        return path, filename




