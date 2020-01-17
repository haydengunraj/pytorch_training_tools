import os
import json
import torch

from ..validation.metrics import get_mode


class CheckpointSaver:
    def __init__(self, checkpoint_dir, model, optimizer, scheduler, save_interval=1, last_epoch=None,
                 model_prefix='model_weights_epoch_', optimizer_prefix='optimizer_state_epoch_',
                 scheduler_prefix='scheduler_state_epoch_', save_best_only=True, metric='loss'):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, 'checkpoint.json')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_interval = save_interval
        self.last_epoch = last_epoch
        self.prefixes = {
            'model': model_prefix,
            'optimizer': optimizer_prefix,
            'scheduler': scheduler_prefix
        }
        self.save_best_only = save_best_only
        self.save_paths = None
        self.checkpoint_dict = None
        self.metric = metric
        self.last_metric_val = None
        self.mode = get_mode(metric)

    def update_checkpoint(self, epoch, step, metric_vals=None):
        save_paths, filenames = self._make_save_paths(epoch)
        if not epoch % self.save_interval and epoch != self.last_epoch:
            if self.save_best_only:
                if metric_vals is None or self.metric not in metric_vals:
                    print('Metric {} not found in metric vals, defaulting to always saving'.format(self.metric))
                    self.save_best_only = False
                    self.update_checkpoint(epoch, step, metric_vals)
                elif self.is_improved(metric_vals[self.metric]):
                    self._save_states(save_paths, filenames, epoch, step)

                    if self.last_metric_val is None:
                        save_str = '{} improved from {} to {:.3f}, saving checkpoint'
                    else:
                        save_str = '{} improved from {:.3f} to {:.3f}, saving checkpoint'
                    print(save_str.format(self.metric.capitalize(), self.last_metric_val, metric_vals[self.metric]))
                    self.last_metric_val = metric_vals[self.metric]

                    if self.save_paths is not None:
                        self._remove_previous_checkpoint()
                    self.save_paths = save_paths
                else:
                    print('{} worsened from {:.3f} to {:.3f}, not saving checkpoint'.format(
                        self.metric.capitalize(), self.last_metric_val, metric_vals[self.metric]))
            else:
                self._save_states(save_paths, filenames, epoch, step)
                print('Saved checkpoint at epoch {}, step {}'.format(epoch, step))
        else:
            self._save_states(save_paths, filenames, epoch, step)
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

    def _save_states(self, save_paths, filenames, epoch, step):
        torch.save(self.model.state_dict(), save_paths['model'])
        torch.save(self.optimizer.state_dict(), save_paths['optimizer'])
        if self.scheduler is not None:
            torch.save(self.scheduler.state_dict(), save_paths['scheduler'])
        self._update_checkpoint_file(filenames, epoch, step)

    def _remove_previous_checkpoint(self):
        for save_path in self.save_paths.values():
            try:
                os.unlink(save_path)
            except FileNotFoundError:
                print('Could not delete state file at {}'.format(save_path))
                pass

    def _update_checkpoint_file(self, filenames, epoch, step):
        self.checkpoint_dict = {
            'model_state': filenames['model'],
            'optimizer_state': filenames['optimizer'],
            'scheduler_state': filenames['scheduler'] if self.scheduler is not None else None,
            'epoch': str(epoch),
            'step': str(step)
        }
        self._write_checkpoint_file()

    def _write_checkpoint_file(self):
        with open(self.checkpoint_file, 'w') as f:
            json.dump(self.checkpoint_dict, f)

    def _make_save_paths(self, epoch):
        save_paths, filenames = {}, {}
        postfix = '{}.pth'.format(epoch)
        for component, prefix in self.prefixes.items():
            filenames[component] = prefix + postfix
            save_paths[component] = os.path.join(self.checkpoint_dir, filenames[component])
        return save_paths, filenames




