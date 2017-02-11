from pyalgotrade.broker import BaseBrokerImpl
from pyalgotrade.broker import  backtesting
from pyalgotrade import logger
from pyalgotrade.broker import Order

class PytradeBroker(BaseBrokerImpl):
    LOGGER_NAME = "pytrade.broker"

    def __init__(self, feed, cash=None, orders=None, shares=None):
        super(PytradeBroker, self).__init__(cash, feed)

        self.initializeActiveOrders(orders)
        self.initializeShares(shares)
        self.setLogger(logger.getLogger(PytradeBroker.LOGGER_NAME))

    def initializeActiveOrders(self, activeOrders):
        self._activeOrders = activeOrders
        self._nextOrderId = max(activeOrders.keys()) + 1 if len(activeOrders) > 0 else 1

    def initializeShares(self, shares):
        self._shares = shares

    def getNextOrderIdWithoutIncrementing(self):
        return self._nextOrderId

    def getOrderById(self, orderId):
        return self._activeOrders[orderId]

    def getActiveMarketOrders(self):
        return [o for o in self.getActiveOrders() if o.getType()==Order.Type.MARKET]

    def getStopOrdersToConfirm(self):
        bars = self.getFeed().getCurrentBars()
        return [o for o in self.getActiveOrders() if o.getType() == Order.Type.STOP and o.getStopPrice()>=bars.getBar(instrument=o.getInstrument())]

    def getActiveStopOrders(self):
        return [o for o in self.getActiveOrders() if o.getType() == Order.Type.STOP]

    def confirmOrder(self, order, bar):
        self.getLogger().debug("Processing order %s " % (order.getId()))

        self.acceptOrder(bar.getDateTime(), order)
        quantity = order.getQuantity()
        price = bar.getOpen()
        commission=10
        cost, sharesDelta, resultingCash = self.calculateCostSharesDeltaAndResultingCash(order, price, quantity, commission)

        if resultingCash >= 0:
            self.handleOrderExecution(order, bar.getDateTime(), price, quantity, commission, resultingCash, sharesDelta)
            return True
        else:
            self.getLogger().debug("Not enough cash to fill %s order [%s] for %s share/s" % (
                order.getInstrument(),
                order.getId(),
                order.getRemaining()
            ))
            return False

    def createMarketOrder(self, action, instrument, quantity, onClose=False):
        return backtesting.MarketOrder(action, instrument, quantity, onClose, self.getInstrumentTraits(instrument))

    def createStopOrder(self, action, instrument, stopPrice, quantity):
        return backtesting.StopOrder(action, instrument, stopPrice, quantity, self.getInstrumentTraits(instrument))