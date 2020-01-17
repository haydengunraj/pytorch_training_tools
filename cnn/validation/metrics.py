import torch


def accuracy(outputs, labels):
    _, idx = torch.max(outputs, dim=1)
    acc = (idx == labels).sum().float()/idx.size(0)
    return acc


def confusion_matrix(outputs, labels):
    n_classes = outputs.size(1)
    _, idx = torch.max(outputs, dim=1)
    conf_mat = torch.zeros((n_classes, n_classes), dtype=torch.long)
    for i, j in zip(labels, idx):
        conf_mat[i.item(), j.item()] += 1
    return conf_mat


METRICS = {
    'accuracy': {
        'function': accuracy,
        'mode': 'maximize'
    },
    'confusion': {
        'function': confusion_matrix,
        'mode': None
    },
    'loss': {
        'function': None,
        'mode': 'minimize'
    }
}


def get_metrics(metric_names):
    metrics = {}
    for metric_name in metric_names:
        metric = METRICS.get(metric_name)
        if metric is None:
            raise ValueError('Unrecognized metric: {}'.format(metric_name))
        metrics[metric_name] = metric['function']
    return metrics


def get_mode(metric_name):
    metric = METRICS.get(metric_name)
    if metric is None:
        raise ValueError('Unrecognized metric: {}'.format(metric_name))
    return metric['mode']

