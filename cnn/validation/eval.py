import torch
import torch.utils.data as data

from .metrics import get_metrics


def eval_pass(model, dataset, batch_size=1, device='cpu', num_workers=1):
    model.eval()
    eval_loader = data.DataLoader(dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers)
    with torch.no_grad():
        outputs = torch.zeros((len(dataset), len(dataset.classes)))
        labels = torch.zeros((len(dataset)), dtype=torch.long)
        outputs, labels = outputs.to(device), labels.to(device)
        for batch_num, data_batch in enumerate(eval_loader):
            batch, lab = data_batch
            batch, lab = batch.to(device), lab.to(device)
            out = model.forward(batch)
            outputs[batch_num*batch_size:(batch_num + 1)*batch_size, :] = out
            labels[batch_num*batch_size:(batch_num + 1)*batch_size] = lab
    model.train()

    return outputs, labels


class Evaluator:
    """Wrapper class for eval"""
    def __init__(self, model, dataset, metrics, writer=None, batch_size=1, num_workers=1):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.writer = writer
        self.metrics = get_metrics(metrics)
        self.num_workers = num_workers

    def eval(self, step, device='cpu'):
        print('Starting eval at step {}'.format(step))
        outputs, labels = eval_pass(self.model, self.dataset, batch_size=self.batch_size,
                                    device=device, num_workers=self.num_workers)
        metrics = {name: metric(outputs, labels).item() for name, metric in self.metrics.items()}
        if self.writer is not None:
            for metric_name, metric_val in metrics.items():
                self.writer.add_scalar('val/val_{}'.format(metric_name), metric_val, step)
        return metrics
