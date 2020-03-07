from ..utils import import_submodule


def get_trainer(config):
    trainer_type = config.pop('type')
    module = import_submodule(__name__, trainer_type)
    if module is None:
        raise ValueError('Unrecognized trainer type: ' + trainer_type)
    trainer = module.get_trainer(config)
    return trainer
