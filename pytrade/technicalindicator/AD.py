from pytrade.technicalindicator import TechnicalIndicatorCalculator
from talib import STOCHF
import numpy as np

class ADOSCCalculator(TechnicalIndicatorCalculator):
    def __init__(self, size):
        self.__size = size

    def calculate(self, feed, instrument):
        high = np.array(feed[instrument].getHighDataSeries())
        low = np.array(feed[instrument].getLowDataSeries())
        close = np.array(feed[instrument].getCloseDataSeries())
        volume = np.array(feed[instrument].getVolumeDataSeries())
        result = STOCHF(high=high, low=low, close=close)
        return result[1][-1]