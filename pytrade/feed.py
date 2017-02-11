from pyalgotrade.barfeed import sqlitefeed
from pyalgotrade import bar

class DynamicFeed(sqlitefeed.Feed):
    """Class for a dynamic feed, with flexibility to allow a feed for any day, at any time - no need for the feeds be serial.
    Based on a sqlitedb which should be feed with stock data from google finance.

     :param dbFilePath: The file with the sqlitedb. Will create a new DB if the file does not exists
     :type dbFilePath: :string.
     :param instruments: The instruments to be loaded from DB
     :type instruments: list
     :param fromDateTime: the initial date to load bars
     :type fromDateTime: datetime
     :param toDateTime: the final date to load bars
     :type toDateTime: datetime
     :param maxLen: The maxLen for the data series
     :type maxLen: :int.

     """

    def __init__(self, dbFilePath, instruments, fromDateTime=None, toDateTime=None, maxLen=180):
        super(DynamicFeed, self).__init__(dbFilePath, bar.Frequency.DAY, maxLen=maxLen)
        for instrument in instruments:
            self.loadBars(instrument, fromDateTime=fromDateTime, toDateTime=toDateTime)

    def nextEvent(self):
        dateTime, values = self.getNextValuesAndUpdateDS()
        return dateTime is not None

    def positionFeed(self, day):
        """
            Position the feed at the specified day for all the registered instruments: the field will be ready to process the specified day
             :param day: The day at wich the feed will be positioned
             :type day: datetime
        """

        self.reset()
        while not self.eof() and self.peekDateTime() <= day:
            self.nextEvent()

    def getAllDays(self):
        days = []
        self.reset()
        while not self.eof():
            self.nextEvent()
            days.append(self.peekDateTime())

        return filter(None, days)

    def dispatchWithoutIncrementingDate(self):
        dateTime, values = self.getCurrentValues()
        if dateTime is not None:
            self.getNewValuesEvent().emit(dateTime, values)
        return dateTime is not None