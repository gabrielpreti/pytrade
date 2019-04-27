# -*- coding: utf-8 -*-
from pyalgotrade import strategy
from pyalgotrade.broker import Order
import abc
import logging

class TradingSystem(strategy.BaseStrategy):
    def __init__(self, feed, broker, debugMode=True, tradingAlgorithm=None):
        super(TradingSystem, self).__init__(feed, broker)
        self.setUseEventDateTimeInLogs(True)
        self.__setDebugMode(debugMode)

        self.__tradingAlgorithm = tradingAlgorithm

    def setAlgorithm(self, tradingAlgorithm):
        self.__tradingAlgorithm = tradingAlgorithm

    def getAlgorithm(self):
        return self.__tradingAlgorithm

    def __setDebugMode(self, debugOn):
        """Enable/disable debug level messages in the strategy and backtesting broker.
        This is enabled by default."""
        level = logging.DEBUG if debugOn else logging.INFO
        self.getLogger().setLevel(level)
        self.getBroker().getLogger().setLevel(level)

    def onOrderUpdated(self, order):
        assert order.getType() == Order.Type.MARKET or order.getType() == Order.Type.STOP

        if order.getType() == Order.Type.STOP:
            return

        assert order.getAction() == Order.Action.BUY or order.getAction() == Order.Action.SELL

        instrument = order.getInstrument()
        if order.getAction() == Order.Action.BUY and (order.getState() == Order.State.FILLED or order.getState() == Order.State.PARTIALLY_FILLED):
            shares = self.getBroker().getShares(instrument)
            self.stopOrder(instrument=instrument, stopPrice=order.stopLossValue, quantity=(-1*shares), goodTillCanceled=True, allOrNone=True)
        elif order.getAction() == Order.Action.SELL and order.getState() == Order.State.ACCEPTED:
            #verify if there was a colision between an sell order and a stop order;
            activeOrders = self.getBroker().getActiveOrders(instrument=instrument)
            if len(activeOrders) == 1 and activeOrders[0].getId() == order.getId():
                self.warning("Collision between stop loss and sell condition submitted at %s" % (order.getSubmitDateTime())) #I could solve that if, before explicitly exiting a position, I verified that the trade was proffitable. That way, the only way to exit a proffitable trade would be via stop loss, so no collisions would happen.
                self.getBroker().cancelOrder(order)
        elif order.getAction() == Order.Action.SELL and order.getState() == Order.State.FILLED:
            stopOrder = [ o for o in self.getBroker().getActiveOrders(instrument=instrument) if o.getType()==Order.Type.STOP][0]
            assert stopOrder.getType() == Order.Type.STOP
            self.getBroker().cancelOrder(stopOrder)

    def isOpenPosition(self, instrument):
        return instrument in self.getBroker().getActiveInstruments()

    def enterPosition(self, instrument, quantity, stopLossValue):
        assert quantity > 0;
        assert  not self.isOpenPosition(instrument)

        self.debug("Order to buy %s shares of %s at %s" % (quantity, instrument, self.getBroker().getCurrentDateTime()))
        order = self.marketOrder(instrument=instrument, quantity=quantity, goodTillCanceled=False, allOrNone=True)
        order.stopLossValue = stopLossValue #o ideal seria um ter um novo tipo de ordem pra preencher esse valor.

    def exitPosition(self, instrument):
        assert instrument in self.getBroker().getActiveInstruments()
        qty = self.getBroker().getShares(instrument)
        assert qty>0

        self.debug("Order to sell %s shares of %s at %s" % (qty, instrument, self.getBroker().getCurrentDateTime()))
        self.marketOrder(instrument=instrument, quantity=(-1*qty))

    def onBarsImpl(self, bars, instrument):
        self.__tradingAlgorithm.onBars(bars, instrument)

        if (not self.__tradingAlgorithm.shouldAnalyze(bars, instrument)):
            self.debug("Skipping stock %s at date %s" % (len(bars.getInstruments()), bars.getDateTime()))
            return

        bar = bars.getBar(instrument)
        if self.isOpenPosition(instrument):  # open position
            if self.__tradingAlgorithm.shouldSellStock(bar, instrument):
                self.exitPosition(instrument)
        elif self.__tradingAlgorithm.shouldBuyStock(bar, instrument):
            size = self.__tradingAlgorithm.calculateEntrySize(bar, instrument)
            stopLoss = self.__tradingAlgorithm.calculateStopLoss(bar, instrument)
            self.enterPosition(instrument, size, stopLoss)

    def onBars(self, bars):
        assert self.__tradingAlgorithm is not None, "Algorithm not attached."
        self.info("Processing date %s" % (bars.getDateTime()))
        for instrument in bars.getInstruments():
            self.onBarsImpl(bars, instrument)


class TradingAlgorithm(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, feed, broker, technicalIndicators=None):
        self._feed = feed
        self._broker = broker
        self._technicalIndicators = technicalIndicators

    def onBars(self, bars, instrument):
        if self._technicalIndicators is None:
            return
        for ti in self._technicalIndicators.values():
            ti.onBars(bars, instrument)

    def getBroker(self):
        return self._broker

    def getTechnicalIndicators(self):
        return self._technicalIndicators if self._technicalIndicators is not None else {}

    @abc.abstractmethod
    def shouldAnalyze(self, bar, instrument):
        """
        :param dateTime: datetime of the event.
        :type dateTime: dateTime.datetime.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def shouldSellStock(self, bar, instrument):
        raise NotImplementedError()

    @abc.abstractmethod
    def shouldBuyStock(self, bar, instrument):
        raise NotImplementedError()

    @abc.abstractmethod
    def calculateEntrySize(self, bar, instrument):
        raise  NotImplementedError()

    @abc.abstractmethod
    def calculateStopLoss(self, bar, instrument):
        raise  NotImplementedError()