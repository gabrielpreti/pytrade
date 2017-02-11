from pytrade.technicalindicator import TechnicalIndicatorCalculator
from talib import RSI
import numpy as np

class RSICalculator(TechnicalIndicatorCalculator):
    def __init__(self, rsiPeriod):
        self.__rsiPeriod = rsiPeriod

    def calculate(self, feed, instrument):
        closevalues = np.array(feed[instrument].getCloseDataSeries())
        return RSI(closevalues, self.__rsiPeriod)[-1] if len(closevalues)>self.__rsiPeriod else None