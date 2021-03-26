

class VehicleModel:
    """
    Super class for making vehicle models that all have the same interface
    """
    X = 0.
    Y = 0.
    yaw = 0.

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def set(self, **kwargs):
        self.__dict__.update(kwargs)

    def update(self, *args, **kwargs):
        raise NotImplementedError()

    def observe(self, *args, **kwargs):
        raise NotImplementedError()
