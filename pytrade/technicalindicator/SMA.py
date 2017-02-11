from pytrade.technicalindicator import TechnicalIndicatorCalculator
from talib import SMA
import numpy as np

SMA_CODE = "SMA"
class SMACalculator(TechnicalIndicatorCalculator):
    def __init__(self, size):
        self.__size = size

    def calculate(self, feed, instrument):
        closevalues = np.array(feed[instrument].getCloseDataSeries())
        return SMA(closevalues, self.__size)[-1] if len(closevalues)>self.__size else None