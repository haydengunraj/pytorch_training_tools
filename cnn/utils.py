from importlib import import_module


def map_inputs(input_keys, data_dict):
    """Maps data with input keys"""
    return [data_dict[key] for key in input_keys]


def map_outputs(output_keys, outputs):
    """Maps data with output keys"""
    if not isinstance(outputs, tuple):
        outputs = outputs,
    return {key: output for key, output in zip(output_keys, outputs)}


def import_submodule(package, name):
    """Import utility for submodules"""
    return import_module('.' + name, package=package)


def get_component(package, name, config, additional_modules=()):
    """Import and initialization utility"""
    try:
        module = import_submodule(package, name)
        component = module.create(config)
    except ImportError:
        component = None
        for module in additional_modules:
            component = getattr(module, name)
            if component is not None:
                break
        if component is None:
            ValueError('Unrecognized component type: ' + name)
        component = component(**config)
    return component


class Wrapper:
    """Base class for wrapping objects"""
    def __init__(self, wrapped_object):
        self.object = wrapped_object

    def __getattr__(self, attr):
        return getattr(self.__dict__['object'], attr)


class CallableWrapper(Wrapper):
    """Wrapper for callable objects"""
    def __init__(self, callable_object, input_keys, output_keys):
        super().__init__(callable_object)
        self.input_keys = input_keys
        self.output_keys = output_keys

    def __call__(self, data_dict):
        inputs = map_inputs(self.input_keys, data_dict)
        outputs = self.object(*inputs)
        output_dict = map_outputs(self.output_keys, outputs)
        return output_dict


class DatasetWrapper(Wrapper):
    """Wrapper for dataset objects"""
    def __init__(self, dataset, output_keys):
        super().__init__(dataset)
        self.output_keys = output_keys

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state

    def __getitem__(self, index):
        return map_outputs(self.output_keys, self.object[index])

    def __len__(self):
        return len(self.object)
