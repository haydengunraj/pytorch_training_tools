from ..utils import map_inputs, map_outputs


class LossWrapper:
    def __init__(self, loss_func, input_map, output_keys):
        self.loss_func = loss_func
        self.input_map = input_map
        self.output_keys = output_keys

    def __getattr__(self, attr):
        return getattr(self.loss_func, attr)

    def __call__(self, input_dict, output_dict):
        loss_inputs = map_inputs(self.input_map, input_dict)
        loss_outputs = self.loss_func(**loss_inputs)
        output_dict = map_outputs(self.output_keys, loss_outputs)
        return output_dict
