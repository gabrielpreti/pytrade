import unittest
from pyalgotrade.tools import googlefinance
from pytrade.feed import DynamicFeed
from pytradeapi import PytradeApi
from pytrade.algorithms.donchianchannels import DonchianTradingAlgorithm
import pytz, datetime

class ApiIntegrationTests(unittest.TestCase):
    db = "./sqliteddb"
    codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3",
             "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3",
             "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4",
             "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3",
             "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
    csvStorage = "./googlefinance"
    donchianEntry = 9
    donchianExit = 26
    riskFactor = 0.05
    initialCash = 10000
    maxLen = int(donchianExit * 1.4)
    username = "gabriel"

    @classmethod
    def setUpClass(cls):
        feed = DynamicFeed(cls.db, cls.codes)
        days = feed.getAllDays()
        if len(days) == 247:
            return

        rowFilter = lambda row: row["Close"] == "-" or row["Open"] == "-" or row["High"] == "-" or row["Low"] == "-" or \
                                row["Volume"] == "-"
        googleFeed = googlefinance.build_feed(cls.codes, 2014, 2014, storage=cls.csvStorage, skipErrors=True,
                                              rowFilter=rowFilter)
        feed = DynamicFeed(cls.db, cls.codes, maxLen=10)
        feed.getDatabase().addBarsFromFeed(googleFeed)

    def testApiDonchianAlgorithm2014WithSQLiteDataProvider(self):
        feed = DynamicFeed(self.db, self.codes, maxLen=self.maxLen)
        days = feed.getAllDays()

        api = PytradeApi(dbfilepah=self.db)
        api.reinitializeUser(username=self.username, cash=self.initialCash)
        tradingAlgorithmGenerator = lambda feed, broker: DonchianTradingAlgorithm(feed, broker, self.donchianEntry, self.donchianExit, self.riskFactor)

        for i in range(len(days)):
            day = days[i]
            api = PytradeApi(dbfilepah=self.db, username=self.username, tradingAlgorithmGenerator=tradingAlgorithmGenerator, codes=None, date=day, maxlen=self.maxLen, debugmode=False)
            api.executeAnalysis()
            api.persistData()

            if i == (len(days) - 1):
                continue

            day = days[i + 1]
            api = PytradeApi(dbfilepah=self.db, username=self.username, tradingAlgorithmGenerator=tradingAlgorithmGenerator, codes=None, date=day, maxlen=self.maxLen, debugmode=False)

            for order in api.getActiveMarketOrders() + api.getStopOrdersToConfirm():
                bar = api.getCurrentBarForInstrument(order.getInstrument())
                if bar is None:
                    continue

                if not api.confirmOrder(order, bar.getDateTime(), order.getQuantity(), bar.getOpen(), 10):
                    api.cancelOrder(order)

            api.persistData()

        self.assertEqual(api.getEquity(), 36922.16)

    def testApiDonchianAlgorithmWithSpecificDatesAndSQLiteDataProvider(self):
        utc = pytz.utc
        specificdays = [
            utc.localize(datetime.datetime(2014, 2, 7)),
            utc.localize(datetime.datetime(2014, 2, 11)),
            utc.localize(datetime.datetime(2014, 9, 18)),
            utc.localize(datetime.datetime(2014, 10, 23)),
            utc.localize(datetime.datetime(2014, 10, 28)),
            utc.localize(datetime.datetime(2014, 12, 29))]

        feed = DynamicFeed(self.db, self.codes, maxLen=self.maxLen)
        alldays = feed.getAllDays()

        api = PytradeApi(dbfilepah=self.db)
        api.reinitializeUser(username=self.username, cash=self.initialCash)
        tradingAlgorithmGenerator = lambda feed, broker: DonchianTradingAlgorithm(feed, broker, self.donchianEntry,
                                                                                  self.donchianExit, self.riskFactor)
        for day in specificdays:
            api = PytradeApi(dbfilepah=self.db, username=self.username, tradingAlgorithmGenerator=tradingAlgorithmGenerator, codes=None, date=day,
                             maxlen=self.maxLen, debugmode=False)
            api.executeAnalysis()
            api.persistData()

            index = alldays.index(day)
            day = alldays[index+1]
            api = PytradeApi(dbfilepah=self.db, username=self.username, tradingAlgorithmGenerator=tradingAlgorithmGenerator, codes=None, date=day,
                             maxlen=self.maxLen, debugmode=False)

            for order in api.getActiveMarketOrders() + api.getStopOrdersToConfirm():
                bar = api.getCurrentBarForInstrument(order.getInstrument())
                if bar is None:
                    continue

                if not api.confirmOrder(order, bar.getDateTime(), order.getQuantity(), bar.getOpen(), 10):
                    api.cancelOrder(order)

            api.persistData()

        self.assertEqual(api.getEquity(), 36922.16)


