from pytrade.technicalindicator import TechnicalIndicatorCalculator
from talib import SMA
import numpy as np

class SuportCalculator(TechnicalIndicatorCalculator):
    def __init__(self, windowSize):
        self.__windowSize = windowSize

    def calculate(self, feed, instrument):
        lowvalues = feed[instrument].getLowDataSeries()
        return min(lowvalues[-(self.__windowSize + 1):-1]) if len(lowvalues)>self.__windowSize else None

class ResistenceCalculator(TechnicalIndicatorCalculator):
    def __init__(self, windowSize):
        self.__windowSize = windowSize

    def calculate(self, feed, instrument):
        highvalues = feed[instrument].getHighDataSeries()
        return max(highvalues[-(self.__windowSize + 1):-1]) if len(highvalues)>self.__windowSize else None