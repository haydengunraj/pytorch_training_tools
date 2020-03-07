from ..utils import import_submodule


def get_evaluator(config):
    evaluator_type = config.pop('type')
    module = import_submodule(__name__, evaluator_type)
    if module is None:
        raise ValueError('Unrecognized evaluator type: ' + evaluator_type)
    evaluator = module.get_evaluator(config)
    return evaluator
