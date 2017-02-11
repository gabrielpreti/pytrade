import abc

class DataProvider(object):
    def __init__(self):
        pass

    @abc.abstractmethod
    def loadCash(self, user):
        raise NotImplementedError()

    @abc.abstractmethod
    def loadShares(self, user):
        raise NotImplementedError()

    @abc.abstractmethod
    def loadOrders(self, user):
        raise NotImplementedError()

    @abc.abstractmethod
    def persistCash(self, user, cash):
        raise NotImplementedError()

    @abc.abstractmethod
    def persistShares(self, user, shares):
        raise NotImplementedError()

    @abc.abstractmethod
    def persistOrders(self, user, orders):
        raise NotImplementedError()
