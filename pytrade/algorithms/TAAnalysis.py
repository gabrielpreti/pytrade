from pytrade.base import TradingAlgorithm
import math

RSI_CODE = "RSI"
LONGSMA_CODE = "LONGSMA"
SHORTSMA_CODE = "SHORTSMA"
SUPPORT_CODE = "SUPPORT"
RESISTENCE_CODE = "RESISTENCE"
AD_CODE = "AD"

class TAAnalysisTradingAlgorithm(TradingAlgorithm):
    def __init__(self, feed, broker, technicalIndicators, riskFactor):
        super(TAAnalysisTradingAlgorithm, self).__init__(feed, broker, technicalIndicators)
        self.__riskFactor = riskFactor


    def shouldAnalyze(self, bar, instrument):
        for ta in self._technicalIndicators.values():
            if ta[instrument][-1] is None:
                return False
        return True

    def shouldBuyStock(self, bar, instrument):
        longSmaValue = self._technicalIndicators[LONGSMA_CODE][instrument][-1]
        shortSmaValue = self._technicalIndicators[SHORTSMA_CODE][instrument][-1]
        resistencevalue = self._technicalIndicators[RESISTENCE_CODE][instrument][-1]

        shortSmaSerie = self._technicalIndicators[SHORTSMA_CODE][instrument]
        longSmaSerie = self._technicalIndicators[LONGSMA_CODE][instrument]
        smaCondition = shortSmaSerie[-3]<longSmaSerie[-3] and shortSmaSerie[-2]>longSmaSerie[-2] and shortSmaSerie[-1]>longSmaSerie[-1] and (shortSmaSerie[-2]-longSmaSerie[-2])<(shortSmaSerie[-1]-longSmaSerie[-1])
        return smaCondition and (shortSmaValue/longSmaValue)>=1.01

        #return bar.getVolume()>100000 and (bar.getClose()>resistencevalue and (shortsma[-1]/longsma[-1])>=1.15)
        # return (shortSmaValue / longSmaValue) >= 1.1 and (bar.getClose() >= resistencevalue or rsiValue<=20) and rsiValue<=70

    def shouldSellStock(self, bar, instrument):
        supportvalue = self._technicalIndicators[SUPPORT_CODE][instrument][-1]

        shortSmaSerie = self._technicalIndicators[SHORTSMA_CODE][instrument]
        longSmaSerie = self._technicalIndicators[LONGSMA_CODE][instrument]
        smaCondition = shortSmaSerie[-3]>longSmaSerie[-3] and shortSmaSerie[-2]<longSmaSerie[-2] and shortSmaSerie[-1]<longSmaSerie[-1] and (longSmaSerie[-2]-shortSmaSerie[-2])<(longSmaSerie[-1]-shortSmaSerie[-1])

        return smaCondition or bar.getClose()<supportvalue

    def calculateEntrySize(self, bar, instrument):

        totalCash = self.getBroker().getEquity()
        closeValue = bar.getClose()
        stopLossPoint = self.calculateStopLoss(bar, instrument)

        return max(1, math.floor ( (totalCash * self.__riskFactor) / (closeValue - stopLossPoint) ))

    def calculateStopLoss(self, bar, instrument):
        return self._technicalIndicators[SUPPORT_CODE][instrument][-1]