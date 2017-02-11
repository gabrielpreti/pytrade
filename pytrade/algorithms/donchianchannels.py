import math

from pytrade.base import TradingAlgorithm


class DonchianChannel:
    def __init__(self, feed, instrument, entrySize, exitSize):
        self.__feed = feed
        self.__instrument = instrument
        self.__entrySize = entrySize
        self.__exitSize = exitSize

    def getMinDonchian(self):
        return self.__feed[self.__instrument].getLowDataSeries()[-(self.__exitSize+1):-1]

    def getMaxDonchian(self):
        return self.__feed[self.__instrument].getHighDataSeries()[-(self.__entrySize+1):-1]

    def entryValue(self):
        return max(self.getMaxDonchian())

    def exitValue(self):
        return min(self.getMinDonchian())

    def isReady(self):
        return len(self.getMinDonchian())>=self.__exitSize and len(self.getMaxDonchian())>=self.__entrySize

class DonchianTradingAlgorithm(TradingAlgorithm):
    def __init__(self, feed, broker, donchianEntrySize, donchianExitSize, riskFactor):
        super(DonchianTradingAlgorithm, self).__init__(feed, broker)
        self.__riskFactor = riskFactor

        self.__donchians = {}
        for instrument in feed.getRegisteredInstruments():
            self.__donchians[instrument] = DonchianChannel(feed, instrument, donchianEntrySize, donchianExitSize)

    def shouldAnalyze(self, bar, instrument):
        return self.__donchians[instrument].isReady()

    def shouldBuyStock(self, bar, instrument):
        return bar.getVolume()>10000000 and bar.getClose() > self.__donchians[instrument].entryValue()

    def shouldSellStock(self, bar, instrument):
        return bar.getClose() < self.__donchians[instrument].exitValue()


    def calculateEntrySize(self, bar, instrument):

        totalCash = self.getBroker().getEquity()
        closeValue = bar.getClose()
        stopLossPoint = self.calculateStopLoss(bar, instrument)

        return math.floor ( (totalCash * self.__riskFactor) / (closeValue - stopLossPoint) )

    def calculateStopLoss(self, bar, instrument):
        return self.__donchians[instrument].exitValue()

    def getMinDonchian(self, instrument):
        return self.__donchians[instrument].getMinDonchian()

    def getMaxDonchian(self, instrument):
        return self.__donchians[instrument].getMaxDonchian()