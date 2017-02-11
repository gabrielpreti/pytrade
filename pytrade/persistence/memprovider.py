from pytrade.persistence import DataProvider

class MemoryDataProvider(DataProvider):
    def __init__(self):
        super(MemoryDataProvider, self).__init__()
        self.__cash = None
        self.__shares = {}
        self.__orders = {}

    def loadCash(self, user=None):
        return self.__cash

    def loadShares(self, user=None):
        return self.__shares

    def loadOrders(self, user=None):
        return self.__orders

    def persistCash(self, user=None, cash=None):
        assert cash is not None
        self.__cash = cash

    def persistShares(self, user=None, shares=None):
        assert shares is not None
        self.__shares = shares

    def persistOrders(self, user=None, orders=None):
        assert orders is not None
        self.__orders = orders
