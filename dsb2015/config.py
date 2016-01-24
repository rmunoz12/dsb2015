class Config(object):
    """
    Stores path options. Note that all path attributes are expected to end with
    a slash `/`.
    """
    def __init__(self, **kwargs):
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __repr__(self):
        return str(self.__dict__)


