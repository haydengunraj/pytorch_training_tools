from ..utils import get_component


def get_evaluator(config):
    evaluator_type = config.pop('type')
    evaluator = get_component(__name__, evaluator_type, config)
    return evaluator
