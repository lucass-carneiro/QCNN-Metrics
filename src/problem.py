import abc


class Problem(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def cost(self, node, weights, data, N):
        pass
