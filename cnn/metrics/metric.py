MAXIMIZE_MODE = 'maximize'
MINIMIZE_MODE = 'minimize'


class Metric:
    def update(self, data_dict):
        """Update running values"""
        raise NotImplementedError

    def reset(self):
        """Reset running values"""
        raise NotImplementedError

    def value(self):
        """Compute final metric value"""
        raise NotImplementedError
