from ..utils import get_component, import_submodule


def get_metrics(metric_list):
    metric_dict = {}
    for config in metric_list:
        metric_type = config.pop('type')
        metric = get_component(__name__, metric_type, config)
        metric_dict[metric_type] = metric
    return metric_dict


def get_mode(metric_type):
    try:
        module = import_submodule(__name__, metric_type)
        mode = module.MODE
    except ImportError:
        raise ValueError('Unrecognized component type: ' + metric_type)
    return mode
