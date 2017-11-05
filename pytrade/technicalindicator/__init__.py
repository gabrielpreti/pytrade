import abc
from pyalgotrade import dataseries

class TechnicalIndicator(object):


    def __init__(self, feed, name, calculator, newPlot=True):
        self.__feed = feed
        self.__name = name
        self.__calculator = calculator
        self.__values = {}
        self.__newPlot = newPlot

    def getName(self):
        return self.__name

    def isNewPlot(self):
        return self.__newPlot

    def onBars(self, bars, instrument):
        try:
            values = self.__values[instrument]
        except KeyError:
            values = dataseries.SequenceDataSeries()
            self.__values[instrument] = values

        taValue = self.__calculator.calculate(self.__feed, instrument)
        values.appendWithDateTime(bars.getDateTime(), taValue)

    def __getitem__(self, key):
        return self.__values[key]

    def keys(self):
        return self.__values.keys()

class TechnicalIndicatorCalculator(object):
    __metaclass__ = abc.ABCMeta


    @abc.abstractmethod
    def calculate(self, feed, instrument):
        raise NotImplementedError()