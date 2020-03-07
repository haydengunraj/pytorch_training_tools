def map_inputs(data_dict, input_map):
    """Maps dataset outputs to model inputs"""
    return {input_key: data_dict[data_key] for data_key, input_key in input_map}