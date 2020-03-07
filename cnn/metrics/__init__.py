from ..utils import import_submodule


def get_metrics(metric_list):
    metric_dict = {}
    for config in metric_list:
        metric_type = config.pop('type')
        module = import_submodule(__name__, metric_type)
        if module is None:
            raise ValueError('Unrecognized metric type: ' + metric_type)
        metric = module.get_metric(config)
        metric_dict[metric_type] = metric
    return metric_dict


def get_mode(metric_type):
    module = import_submodule(__name__, metric_type)
    if module is None:
        raise ValueError('Unrecognized metric type: ' + metric_type)
    return module.MODE
