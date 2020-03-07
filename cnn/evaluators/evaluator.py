import torch
import torch.utils.data as data

from ..metrics import get_metrics


def create(config):
    return Evaluator(**config)


class Evaluator:
    """Basic evaluator class

    Args:
        model (nn.Module, CallableWrapper): The model to be evaluated.
        dataset (Dataset, DatasetWrapper): The evaluation dataset.
        metrics (iterable[dict]): A list metric configuration dictionaries
            to be passed to get_metrics.
        loss_func (callable, optional): The loss function to be used for loss
            metrics (defaults to None).
        writer (SummaryWriter, optional): A SummaryWriter for logging data
            (defaults to None).
        batch_size (int, optional): The evaluation batch size (defaults to 1).
        num_workers (int, optional): The number of workers to use for
            loading data (defaults to 1).
        eval_interval: (int, optional): The interval in epochs for which
            eval is performed (defaults to 1).

    Methods:
        eval(): Runs the evaluation process.
        eval_pass(): Runs the cor evaluation loop.
        update_metrics(): Updates running metric values.
        compute_metrics(): Computes final metric values.
        reset_metrics(): Resets all metric values.
    """
    def __init__(self, model, dataset, metrics, loss_func=None, writer=None,
                 batch_size=1, num_workers=1, eval_interval=1):
        self.model = model
        self.dataset = dataset
        self.loss_func = loss_func
        self.metrics = get_metrics(metrics)
        self.writer = writer
        self.eval_interval = eval_interval
        self.batch_size = batch_size
        self.num_workers = num_workers

    def eval(self, epoch, step, device='cpu'):
        if not epoch % self.eval_interval:
            print('\nStarting eval at step {}'.format(step))
            self.eval_pass(device=device)
            metrics = self.compute_metrics()
            self.reset_metrics()
            self._log_and_print_metrics(step, metrics)
            return metrics
        return None

    def eval_pass(self, device='cpu'):
        # Store current model state
        is_training = self.model.training
        self.model.eval()

        # Set up dataloader
        eval_loader = data.DataLoader(self.dataset, batch_size=self.batch_size,
                                      shuffle=False, num_workers=self.num_workers)

        # Run eval loop
        with torch.no_grad():
            for batch_num, data_batch in enumerate(eval_loader):
                data_dict = {key: val.to(device) for key, val in data_batch.items()}
                data_dict.update(self.model(data_dict))
                if self.loss_func is not None:
                    data_dict.update(self.loss_func(data_dict))
                self.update_metrics(data_dict)

        # Restore initial model state
        self.model.train(is_training)

    def update_metrics(self, data_dict):
        for metric in self.metrics:
            metric.update(data_dict)

    def compute_metrics(self):
        return {metric.name: metric.value for metric in self.metrics}

    def reset_metrics(self):
        for metric in self.metrics:
            metric.reset()

    def _log_and_print_metrics(self, step, metrics):
        for metric_name, metric_val in metrics.items():
            if self.writer is not None:
                self.writer.add_scalar('val/val_{}'.format(metric_name), metric_val, step)
            print('{}: {}'.format(metric_name.title(), metric_val), flush=True)
