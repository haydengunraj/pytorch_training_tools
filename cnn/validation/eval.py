import torch
import torch.utils.data as data

from .metrics import get_metrics


# TODO: update the eval process so that not all output needs to be stored at once
class Evaluator:
    """Wrapper class for eval"""
    def __init__(self, model, dataset, metrics, eval_interval=1, output_dtype=torch.float32,
                 label_dtype=torch.float32, writer=None, batch_size=1, num_workers=1):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.writer = writer
        self.metrics = get_metrics(metrics)
        self.num_workers = num_workers
        self.eval_interval = eval_interval
        self.output_dtype = output_dtype
        self.label_dtype = label_dtype

    def eval(self, epoch, step, device='cpu'):
        if not epoch % self.eval_interval:
            print('\nStarting eval at step {}'.format(step))
            outputs, labels = self.eval_pass(device=device)
            metrics = {name: metric(outputs, labels).item() for name, metric in self.metrics.items()}
            for metric_name, metric_val in metrics.items():
                if self.writer is not None:
                    self.writer.add_scalar('val/val_{}'.format(metric_name), metric_val, step)
                print('{}: {}'.format(metric_name.title(), metric_val))
            return metrics
        return None

    def eval_pass(self, device='cpu'):
        self.model.eval()
        eval_loader = data.DataLoader(self.dataset, batch_size=self.batch_size,
                                      shuffle=True, num_workers=self.num_workers)
        with torch.no_grad():
            outputs, labels = None, None
            for batch_num, data_batch in enumerate(eval_loader):
                batch, lab = data_batch
                batch, lab = batch.to(device), lab.to(device)
                out = self.model.forward(batch)

                if outputs is None:
                    outputs = torch.zeros((len(self.dataset), *[dim for dim in out.size()[1:]]), dtype=self.output_dtype)
                    labels = torch.zeros((len(self.dataset), *[dim for dim in lab.size()[1:]]), dtype=self.label_dtype)
                    outputs, labels = outputs.to(device), labels.to(device)
                outputs[batch_num*self.batch_size:(batch_num + 1)*self.batch_size] = out
                labels[batch_num*self.batch_size:(batch_num + 1)*self.batch_size] = lab
        self.model.train()

        return outputs, labels
