from ..utils import map_inputs, map_outputs


class ModelWrapper:
    def __init__(self, model, input_map, output_keys):
        self.model = model
        self.input_map = input_map
        self.output_keys = output_keys

    def forward(self, input_dict):
        """Wraps forward pass of model"""
        model_inputs = map_inputs(self.input_map, input_dict)
        model_outputs = self.model(**model_inputs)
        output_dict = map_outputs(self.output_keys, model_outputs)
        return output_dict

    def __getattr__(self, attr):
        return getattr(self.model, attr)

    def __call__(self, input_dict):
        return self.forward(input_dict)
