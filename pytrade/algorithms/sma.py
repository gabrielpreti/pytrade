import numpy as np
import math
from pytrade.base import TradingAlgorithm
from talib import SMA

class SMATradingAlgorithm(TradingAlgorithm):
    def __init__(self, feed, broker, longsize, shortsize, riskFactor):
        super(SMATradingAlgorithm, self).__init__(feed, broker)
        self.__feed = feed
        self.__brocker = broker
        self.__longsize = longsize
        self.__shortsize = shortsize
        self.__riskFactor = riskFactor

    def shouldAnalyze(self, bar, instrument):
        return len(self.__feed[instrument].getCloseDataSeries()) >= self.__longsize

    def shouldBuyStock(self, bar, instrument):
        closevalues = np.array(self.__feed[instrument].getCloseDataSeries())
        longsma = SMA(closevalues, timeperiod=self.__longsize)
        shortsma = SMA(closevalues, timeperiod=self.__shortsize)

        return bar.getVolume()>10000000 and shortsma[-2]<=longsma[-2] and shortsma[-1]>longsma[-1]

    def shouldSellStock(self, bar, instrument):
        closevalues = np.array(self.__feed[instrument].getCloseDataSeries())
        longsma = SMA(closevalues, timeperiod=self.__longsize)
        shortsma = SMA(closevalues, timeperiod=self.__shortsize)

        return shortsma[-2]>longsma[-2] and shortsma[-1]<=longsma[-1]

    def calculateEntrySize(self, bar, instrument):
        availableCash = self.getBroker().getAvailableCash()
        closeValue = bar.getClose()
        return math.ceil(min(1000, availableCash)/closeValue)

    def calculateStopLoss(self, bar, instrument):
        positionSize = self.calculateEntrySize(bar, instrument)
        closeValue = bar.getClose()
        if positionSize==0:
            return closeValue

        equity = self.getBroker().getEquity()
        return closeValue - (equity*self.__riskFactor)/positionSize