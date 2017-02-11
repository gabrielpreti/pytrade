# from pyalgotrade.tools import googlefinance
# from pyalgotrade.barfeed import sqlitefeed
# import os
# from pyalgotrade import bar
# import pytz, datetime
#
# rowFilter = lambda row: row["Close"] == "-" or row["Open"] == "-" or row["High"] == "-" or row["Low"] == "-" or row["Volume"] == "-"
# instruments = ["PETR4", "PETR3"]
# feed = googlefinance.build_feed(instruments, 2015, 2015, storage="./googlefinance", skipErrors=True, rowFilter=rowFilter)
#
# dbFeed = sqlitefeed.Feed("./sqlitedb", bar.Frequency.DAY, maxLen=None)
# dbFeed.getDatabase().addBarsFromFeed(feed)
#
# #Preciso carregar os bars e percorrer o feed pra alimentar as series
# #Posso abstrair isso em uma classe que, eu passo uma data, ele carrega os bars do DB, percorre o feed pra alimentar as series e ja chega no dia que eu especifiquei com os dias carregados.
# start = datetime.datetime(2015, 1, 5, tzinfo=pytz.UTC)
# end = datetime.datetime(2015, 1, 10, tzinfo=pytz.UTC)
# for instrument in instruments:
#     dbFeed.loadBars(instrument, fromDateTime=start, toDateTime=end)
#
# while dbFeed.dispatch() and dbFeed.getCurrentDateTime() < end:
#     print("passing %s" % (dbFeed.getCurrentDateTime()))
#
#
# dbFeed.getCurrentBars().getBar("PETR4").getDateTime()
# #dbFeed.reset() with this, maybe I don't need to use DB
#
# for instrument in dbFeed.getRegisteredInstruments():
#     series = dbFeed.getDataSeries(instrument)
#     print("%s %s") % (instrument, series.getCloseDataSeries()[0:len(series)])


import pytz, datetime
from pyalgotrade.tools import googlefinance
from pytrade.feed import DynamicFeed




rowFilter = lambda row: row["Close"] == "-" or row["Open"] == "-" or row["High"] == "-" or row["Low"] == "-" or row["Volume"] == "-"
instruments = ["PETR4", "PETR3"]
# googleFeed = googlefinance.build_feed(instruments, 2015, 2015, storage="./googlefinance", skipErrors=True, rowFilter=rowFilter)

dynamicFeed = DynamicFeed("./sqlitedb", instruments, maxLen=10)

dynamicFeed.positionFeed(datetime.datetime(2015, 1, 2, tzinfo=pytz.UTC))

print dynamicFeed.getCurrentDateTime()
for instrument in dynamicFeed.getRegisteredInstruments():
    series = dynamicFeed.getDataSeries(instrument)
    print("%s %s") % (instrument, series.getCloseDataSeries()[0:len(series)])


#2015-01-01
#datetime: None
#PETR4:
    #close value:
    #close series []
#PETR3
    # close value
    # close series []
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 1, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() is None
assert dynamicFeed.getCurrentBars() is None
assert len(dynamicFeed.getDataSeries("PETR4")) == 0
assert len(dynamicFeed.getDataSeries("PETR3")) == 0

#2015-01-02
#datetime: 2015-01-02
#PETR4:
    #close value: 9.36
    #close series [9.36]
#PETR3
    # close value:  9.00
    # close series: [9.00]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 2, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 2, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 9.36
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 9.00
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00]

#2015-01-03
#datetime: 2015-01-02
#PETR4:
# close value: 9.36
    #close series: [9.36]
#PETR3
    # close value:  9.00
    # close series: [9.00]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 3, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 2, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 9.36
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 9.00
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00]

#2015-01-04
#datetime: 2015-01-02
#PETR4:
    # close value: 9.36
    #close series: [9.36]
#PETR3
    # close value:  9.00
    # close series: [9.00]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 4, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 2, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 9.36
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 9.00
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00]

#2015-01-05
#datetime: 2015-01-05
#PETR4:
    #close value:   8.61
    #close series:  [9.36, 8.61]
#PETR3
    # close value:  8.27
    # close series: [9.00, 8.27]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 5, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 5, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 8.61
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36, 8.61]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 8.27
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00, 8.27]

#2015-01-06
#datetime: 2015-01-06
#PETR4:
    #close value:   8.33
    #close series   [9.36, 8.61, 8.33]
#PETR3
    # close value:  8.06
    # close series: [9.00, 8.27, 8.06]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 6, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 6, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 8.33
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36, 8.61, 8.33]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 8.06
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00, 8.27, 8.06]

#2015-01-07
#datetime: 2015-01-07
#PETR4:
    #close value:   8.67
    #close series   [9.36, 8.61, 8.33, 8.67]
#PETR3
    # close value:  8.45
    # close series: [9.00, 8.27, 8.06, 8.45]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 7, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 7, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 8.67
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36, 8.61, 8.33, 8.67]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 8.45
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00, 8.27, 8.06, 8.45]

#2015-01-08
#datetime: 2015-01-08
#PETR4:
    #close value:   9.18
    #close series:  [9.36, 8.61, 8.33, 8.67, 9.18]
#PETR3
    # close value:  9.02
    # close series: [9.00, 8.27, 8.06, 8.45, 9.02]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 8, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 8, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 9.18
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36, 8.61, 8.33, 8.67, 9.18]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 9.02
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00, 8.27, 8.06, 8.45, 9.02]

#2015-01-09
#datetime: 2015-01-09
#PETR4:
    #close value:   9.40
    #close series:  [9.36, 8.61, 8.33, 8.67, 9.18, 9.4]
#PETR3
    # close value:  9.29
    # close series: [9.00, 8.27, 8.06, 8.45, 9.02, 9.29]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 9, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1,9 , tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 9.40
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36, 8.61, 8.33, 8.67, 9.18, 9.4]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 9.29
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00, 8.27, 8.06, 8.45, 9.02, 9.29]

#2015-01-10
#datetime: 2015-01-09
#PETR4:
    #close value:   9.40
    #close series: [9.36, 8.61, 8.33, 8.67, 9.18, 9.4]
#PETR3
    # close value: 9.29
    # close series: [9.00, 8.27, 8.06, 8.45, 9.02, 9.29]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 10, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 9, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 9.40
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36, 8.61, 8.33, 8.67, 9.18, 9.4]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 9.29
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00, 8.27, 8.06, 8.45, 9.02, 9.29]

#2015-01-11
#datetime: 2015-01-09
#PETR4:
    #close value:   9.40
    #close series:  [9.36, 8.61, 8.33, 8.67, 9.18, 9.4]
#PETR3
    # close value:  9.29
    # close series: [9.00, 8.27, 8.06, 8.45, 9.02, 9.29]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 11, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 9, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 9.40
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36, 8.61, 8.33, 8.67, 9.18, 9.4]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 9.29
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00, 8.27, 8.06, 8.45, 9.02, 9.29]

#2015-01-12
#datetime: 2015-01-12
#PETR4:
    #close value:   8.91
    #close series:  [9.36, 8.61, 8.33, 8.67, 9.18, 9.4, 8.91]
#PETR3
    # close value:  8.77
    # close series: [9.00, 8.27, 8.06, 8.45, 9.02, 9.29, 8.77]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 12, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 12, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 8.91
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36, 8.61, 8.33, 8.67, 9.18, 9.4, 8.91]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 8.77
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00, 8.27, 8.06, 8.45, 9.02, 9.29, 8.77]

#2015-01-13
#datetime: 2015-01-13
#PETR4:
    #close value:   9.00
    #close series:  [9.36, 8.61, 8.33, 8.67, 9.18, 9.4, 8.91, 9.00]
#PETR3
    # close value:  8.83
    # close series: [9.00, 8.27, 8.06, 8.45, 9.02, 9.29, 8.77, 8.83]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 13, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 13, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 9.00
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [9.36, 8.61, 8.33, 8.67, 9.18, 9.4, 8.91, 9.00]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 8.83
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.00, 8.27, 8.06, 8.45, 9.02, 9.29, 8.77, 8.83]

#2015-01-16
#datetime: 2015-01-16
#PETR4:
    #close value:   9.44
    #close series:  [8.61, 8.33, 8.67, 9.18, 9.40, 8.91, 9.00, 8,74 9.34, 9.44]
#PETR3
    # close value:  9.23
    # close series: [8.27, 8.06, 8.45, 9.02, 9.29, 8.77, 8.83, 8.50, 9.25,9.23]
dynamicFeed.positionFeed(datetime.datetime(2015, 1, 16, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 1, 16, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 9.44
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [8.61, 8.33, 8.67, 9.18, 9.40, 8.91, 9.00, 8.74, 9.34, 9.44]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 9.23
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [8.27, 8.06, 8.45, 9.02, 9.29, 8.77, 8.83, 8.50, 9.25,9.23]


#2015-12-30
#datetime: 2015-12-30
#PETR4:
    #close value:  6.70
    #close series:  [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70]
#PETR3
    # close value: 8.57
    # close series: [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57]
dynamicFeed.positionFeed(datetime.datetime(2015, 12, 30, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 12, 30, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 6.70
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 8.57
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57]

#2015-12-31
#datetime: 2015-12-30
#PETR4:
    #close value:  6.70
    #close series:  [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70]
#PETR3
    # close value: 8.57
    # close series: [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57]
dynamicFeed.positionFeed(datetime.datetime(2015, 12, 31, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 12, 30, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 6.70
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 8.57
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57]


#2016-1-1
#datetime: 2015-12-30
#PETR4:
    #close value:  6.70
    #close series:  [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70]
#PETR3
    # close value: 8.57
    # close series: [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57]
dynamicFeed.positionFeed(datetime.datetime(2016, 1, 1, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 12, 30, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 6.70
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 8.57
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57]


#2016-1-2
#datetime: 2015-12-30
#PETR4:
    #close value:  6.70
    #close series:  [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70]
#PETR3
    # close value: 8.57
    # close series: [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57]
dynamicFeed.positionFeed(datetime.datetime(2016, 1, 2, tzinfo=pytz.UTC))
assert dynamicFeed.getCurrentDateTime() == datetime.datetime(2015, 12, 30, tzinfo=pytz.UTC)

assert dynamicFeed.getCurrentBars().getBar("PETR4").getClose() == 6.70
assert dynamicFeed.getDataSeries("PETR4").getCloseDataSeries()[0:20] == [7.42, 7.29, 7.20, 7.02, 6.64, 6.79, 6.93, 6.70, 6.69, 6.70]

assert dynamicFeed.getCurrentBars().getBar("PETR3").getClose() == 8.57
assert dynamicFeed.getDataSeries("PETR3").getCloseDataSeries()[0:20] == [9.10, 8.95, 8.82, 8.66, 8.31, 8.54, 8.88, 8.61, 8.57, 8.57]
