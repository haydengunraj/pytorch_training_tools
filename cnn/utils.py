from importlib import import_module


def map_inputs(input_map, input_dict):
    """Maps data with input keys"""
    return {input_key: input_dict[data_key] for data_key, input_key in input_map.items()}


def map_outputs(output_keys, outputs):
    """Maps data with output keys"""
    if not isinstance(outputs, tuple):
        outputs = outputs,
    return {key: output for key, output in zip(output_keys, outputs)}


def import_submodule(package, submodule):
    """Import utility for submodules"""
    try:
        return import_module('.' + submodule, package=package)
    except ImportError:
        return None
