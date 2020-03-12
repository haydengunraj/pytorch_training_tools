MAXIMIZE_MODE = 'maximize'
MINIMIZE_MODE = 'minimize'


class Metric:
    """Base class for all metrics"""
    def __init__(self, name):
        self.name = name

    def update(self, data_dict):
        """Update running values"""
        raise NotImplementedError

    def reset(self):
        """Reset running values"""
        raise NotImplementedError

    def log(self, writer, step, tag_prefix='val/'):
        """Log the current value(s)"""
        raise NotImplementedError

    @property
    def value(self):
        """Compute final metric value"""
        raise NotImplementedError


class LogMetric:
    """Base class for logged metrics with no return value"""
    def __init__(self, name):
        self.name = name

    def update(self, data_dict):
        """Update running values"""
        raise NotImplementedError

    def reset(self):
        """Reset running values"""
        raise NotImplementedError

    def log(self, writer, step, tag_prefix='val/'):
        """Log the current value(s)"""
        raise NotImplementedError

    @property
    def value(self):
        """Compute final metric value"""
        return None
