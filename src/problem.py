import abc


class Problem(metaclass=abc.ABCMeta):
    def __init__(self):
        self.map = None
        self.target = None
        self.f = None

    @abc.abstractmethod
    def cost(self, node, weights, data, N):
        pass

    @abc.abstractmethod
    def get_domain_map(self):
        pass
