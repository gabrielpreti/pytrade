import unittest
from pytrade.feed import DynamicFeed
from datetime import timedelta
from pytrade.broker import PytradeBroker
from pytrade.base import TradingSystem
from pytrade.algorithms.donchianchannels import DonchianTradingAlgorithm
from pyalgotrade.tools import googlefinance
from pytrade.backtesting.backtest import GoogleFinanceBacktest
from pytrade.persistence.memprovider import MemoryDataProvider
from pytrade.persistence.sqliteprovider import SQLiteDataProvider
import pytz, datetime


class BrokerIntegrationTests(unittest.TestCase):
    db = "./sqliteddb"
    codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3",
             "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3",
             "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4",
             "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3",
             "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
    csvStorage="./googlefinance"
    donchianEntry = 9
    donchianExit = 26
    riskFactor = 0.05
    initialCash = 10000
    maxLen = int(donchianExit * 1.4)


    @classmethod
    def setUpClass(cls):
        feed = DynamicFeed(cls.db, cls.codes)
        days = feed.getAllDays()
        if len(days)==247:
            return

        rowFilter = lambda row: row["Close"] == "-" or row["Open"] == "-" or row["High"] == "-" or row["Low"] == "-" or \
                                row["Volume"] == "-"
        googleFeed = googlefinance.build_feed(cls.codes, 2014, 2014, storage=cls.csvStorage, skipErrors=True,
                                               rowFilter=rowFilter)
        feed = DynamicFeed(cls.db, cls.codes, maxLen=10)
        feed.getDatabase().addBarsFromFeed(googleFeed)

    def runDonchianAlgorithm(self, broker, feed, donchianEntry, donchianExit, riskFactor):
        strategy = TradingSystem(feed, broker, debugMode=False)
        strategy.setAlgorithm(DonchianTradingAlgorithm(feed, broker, donchianEntry, donchianExit, riskFactor))
        feed.dispatchWithoutIncrementingDate()
        feed.nextEvent()
        for order in broker.getActiveMarketOrders() + broker.getStopOrdersToConfirm():
            bar = broker.getCurrentBarForInstrument(order.getInstrument())
            if bar is None:
                continue

            if not broker.confirmOrder(order, bar):
                broker.cancelOrder(order)

    def testLiveBrokerDonchianAlgorithmWithSpecificDatesAndSQLiteDataProvider(self):
        username = "gabriel"
        utc = pytz.utc
        days = [
            utc.localize(datetime.datetime(2014, 2, 7)),
            utc.localize(datetime.datetime(2014, 2, 11)),
            utc.localize(datetime.datetime(2014, 9, 18)),
            utc.localize(datetime.datetime(2014, 10, 23)),
            utc.localize(datetime.datetime(2014, 10, 28)),
            utc.localize(datetime.datetime(2014, 12, 31))]

        dataProvider = SQLiteDataProvider(self.db)
        dataProvider.createSchema()
        dataProvider.initializeUser(username, self.initialCash)

        for day in days:
            fromDate = day - timedelta(days=self.maxLen)
            toDate = day + timedelta(days=5)
            feed = DynamicFeed(self.db, self.codes, fromDateTime=fromDate, toDateTime=toDate, maxLen=self.maxLen)
            feed.positionFeed(day)

            broker = PytradeBroker(feed, cash=dataProvider.loadCash(username), orders=dataProvider.loadOrders(username), shares=dataProvider.loadShares(username))
            self.runDonchianAlgorithm(broker, feed, self.donchianEntry, self.donchianExit, self.riskFactor)

            dataProvider.persistCash(username, broker.getAvailableCash())
            dataProvider.persistShares(username, broker.getAllShares())
            dataProvider.persistOrders(username, broker.getAllActiveOrders())

        self.assertEqual(broker.getEquity(), 36922.16)

    def testLiveBrokerDonchianAlgorithm2014WithSQLiteDataProvider(self):
        username = "gabriel"

        feed = DynamicFeed(self.db, self.codes, maxLen=self.maxLen)
        days = feed.getAllDays()

        dataProvider = SQLiteDataProvider(self.db)
        dataProvider.createSchema()
        dataProvider.initializeUser(username, self.initialCash)

        for day in days:
            fromDate = day - timedelta(days=self.maxLen)
            toDate = day + timedelta(days=5)
            feed = DynamicFeed(self.db, self.codes, fromDateTime=fromDate, toDateTime=toDate, maxLen=self.maxLen)
            feed.positionFeed(day)

            broker = PytradeBroker(feed, cash=dataProvider.loadCash(username), orders=dataProvider.loadOrders(username), shares=dataProvider.loadShares(username))
            self.runDonchianAlgorithm(broker, feed, self.donchianEntry, self.donchianExit, self.riskFactor)

            dataProvider.persistCash(username, broker.getAvailableCash())
            dataProvider.persistShares(username, broker.getAllShares())
            dataProvider.persistOrders(username, broker.getAllActiveOrders())

        self.assertEqual(broker.getEquity(), 36922.16)



    def testLiveBrokerDonchianAlgorithm2014WithoutDataProvider(self):
        feed = DynamicFeed(self.db, self.codes, maxLen=self.maxLen)
        days = feed.getAllDays()

        cash = self.initialCash
        shares = {}
        orders = {}
        for day in days:
            fromDate = day - timedelta(days=self.maxLen)
            toDate = day + timedelta(days=5)
            feed = DynamicFeed(self.db, self.codes, fromDateTime=fromDate, toDateTime=toDate, maxLen=self.maxLen)
            feed.positionFeed(day)

            broker = PytradeBroker(feed, cash=cash, orders=orders, shares=shares)
            self.runDonchianAlgorithm(broker, feed, self.donchianEntry, self.donchianExit, self.riskFactor)

            cash = broker.getAvailableCash()
            shares = broker.getAllShares()
            orders = broker.getAllActiveOrders()

        self.assertEqual(broker.getEquity(), 36922.16)

    def testLiveBrokerDonchianAlgorithm2014WithMemoryDataProvider(self):
        feed = DynamicFeed(self.db, self.codes, maxLen=self.maxLen)
        days = feed.getAllDays()

        dataProvider = MemoryDataProvider()
        dataProvider.persistCash(cash=self.initialCash)
        for day in days:
            fromDate = day - timedelta(days=self.maxLen)
            toDate = day + timedelta(days=5)
            feed = DynamicFeed(self.db, self.codes, fromDateTime=fromDate, toDateTime=toDate, maxLen=self.maxLen)
            feed.positionFeed(day)

            broker = PytradeBroker(feed, cash=dataProvider.loadCash(), orders=dataProvider.loadOrders(), shares=dataProvider.loadShares())
            self.runDonchianAlgorithm(broker, feed, self.donchianEntry, self.donchianExit, self.riskFactor)

            dataProvider.persistCash(cash=broker.getAvailableCash())
            dataProvider.persistShares(shares=broker.getAllShares())
            dataProvider.persistOrders(orders=broker.getAllActiveOrders())

        self.assertEqual(broker.getEquity(), 36922.16)

    def testBacktestingDonchianAlgorithm2014(self):
        backtest = GoogleFinanceBacktest(instruments=self.codes, initialCash=self.initialCash, year=2014, debugMode=False,
                                         csvStorage=self.csvStorage)
        backtest.attachAlgorithm(DonchianTradingAlgorithm(backtest.getFeed(), backtest.getBroker(), self.donchianEntry, self.donchianExit, self.riskFactor))
        backtest.run()

        self.assertEqual(backtest.getBroker().getEquity(), 36922.16)