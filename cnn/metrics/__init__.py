from ..utils import get_component

# Keys for evaluator and saver
VALUE_KEY = 'value'
MODE_KEY = 'mode'


def get_metrics(metric_list):
    metrics = []
    for config in metric_list:
        metric_type = config.pop('type')
        metric = get_component(__name__, metric_type, config)
        metrics.append(metric)
    return metrics
