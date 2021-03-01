import abc


class Application(abc.ABC):
    @abc.abstractmethod
    def __init__(self, args, log):
        raise NotImplementedError

    @abc.abstractclassmethod
    def setup_parser(cls, parser):
        raise NotImplementedError

    @abc.abstractmethod
    def execute(self):
        raise NotImplementedError
