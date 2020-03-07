import torch
import torch.utils.data as data

from ..utils import map_inputs
from ..metrics import get_metrics


def get_evaluator(config):
    return Evaluator(**config)


class Evaluator:
    def __init__(self, model, dataset, input_map, metrics, writer=None,
                 batch_size=1, num_workers=1, eval_interval=1):
        self.model = model
        self.dataset = dataset
        self.input_map = input_map
        self.metrics = get_metrics(metrics)
        self.writer = writer
        self.eval_interval = eval_interval
        self.batch_size = batch_size
        self.num_workers = num_workers

    def eval(self, epoch, step, device='cpu'):
        if not epoch % self.eval_interval:
            print('\nStarting eval at step {}'.format(step))
            self.eval_pass(device=device)
            metrics = self.get_metrics()
            self.log_and_print_metrics(step, metrics)
            return metrics
        return None

    def eval_pass(self, device='cpu'):
        self.model.eval()
        eval_loader = data.DataLoader(self.dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=self.num_workers)
        with torch.no_grad():
            for batch_num, data_batch in enumerate(eval_loader):
                data_dict = {key: val.to(device) for key, val in data_batch.items()}
                data_dict.update(self.model(data_dict))
                data_dict.update(self.loss_func(data_dict))

                model_inputs = map_inputs(data_batch, self.input_map)
                data_dict['outputs'] = self.model(**model_inputs)
                self.update_metrics(data_dict)
        self.model.train()

    def update_metrics(self, data_dict):
        for metric in self.metrics:
            metric.update(data_dict)

    def get_metrics(self):
        return {name: metric.value() for name, metric in self.metrics.items()}

    def log_and_print_metrics(self, step, metrics):
        for metric_name, metric_val in metrics.items():
            if self.writer is not None:
                self.writer.add_scalar('val/val_{}'.format(metric_name), metric_val, step)
            print('{}: {}'.format(metric_name.title(), metric_val), flush=True)
