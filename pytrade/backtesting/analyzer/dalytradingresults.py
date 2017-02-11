from pyalgotrade import stratanalyzer
from pyalgotrade import logger
from pyalgotrade import dataseries
from pyalgotrade import broker

class DailyTradingResults(stratanalyzer.StrategyAnalyzer):
    LOGGER_NAME = "MyCustomAnalyzer"

    def __init__(self, maxLen=None):
        self.__logger = logger.getLogger(DailyTradingResults.LOGGER_NAME)
        self.__totalCapital = dataseries.SequenceDataSeries(maxLen=maxLen)
        self.__tradeResults = dataseries.SequenceDataSeries(maxLen=maxLen)
        self.__posTracker = {}
        self.__dailyTradeValues = {}

    def getLogger(self):
        return self.__logger

    def debug(self, msg):
        """Logs a message with level DEBUG on the strategy logger."""
        self.getLogger().debug(msg)

    def info(self, msg):
        """Logs a message with level INFO on the strategy logger."""
        self.getLogger().info(msg)

    def warning(self, msg):
        """Logs a message with level WARNING on the strategy logger."""
        self.getLogger().warning(msg)

    def error(self, msg):
        """Logs a message with level ERROR on the strategy logger."""
        self.getLogger().error(msg)

    def critical(self, msg):
        """Logs a message with level CRITICAL on the strategy logger."""
        self.getLogger().critical(msg)

    def attached(self, strat):
        strat.getBroker().getOrderUpdatedEvent().subscribe(self.__onOrderEvent)

    def beforeOnBars(self, strat, bars):
        barDate = bars.getDateTime()
        self.__totalCapital.appendWithDateTime(barDate, strat.getBroker().getEquity())

        try:
            dailyValue = self.__dailyTradeValues[bars.getDateTime()]
        except KeyError:
            dailyValue = 0;
        self.__tradeResults.appendWithDateTime(bars.getDateTime(), dailyValue)


    def __onOrderEvent(self, broker_, orderEvent):
        # Only interested in filled or partially filled orders.
        if orderEvent.getEventType() not in (broker.OrderEvent.Type.PARTIALLY_FILLED, broker.OrderEvent.Type.FILLED):
            return

        order = orderEvent.getOrder()

        # Update the tracker for this order.
        execInfo = orderEvent.getEventInfo()
        orderDate = execInfo.getDateTime()
        price = execInfo.getPrice()
        quantity = execInfo.getQuantity()
        commission = execInfo.getCommission()
        instrument = order.getInstrument()
        action = order.getAction()
        if action in [broker.Order.Action.BUY, broker.Order.Action.BUY_TO_COVER]:
            self.__posTracker[instrument] = price * quantity
        elif action in [broker.Order.Action.SELL, broker.Order.Action.SELL_SHORT]:
            gain = (price * quantity) - self.__posTracker[instrument] - commission
            del self.__posTracker[instrument]

            try:
                self.__dailyTradeValues[orderDate] = self.__dailyTradeValues[orderDate] + gain
            except KeyError:
                self.__dailyTradeValues[orderDate] = gain
        else:  # Unknown action
            assert (False)





    def getTotalCapitalSeries(self):
        return self.__totalCapital

    def getTradeResults(self):
        return self.__tradeResults
