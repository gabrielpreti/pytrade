import matplotlib
matplotlib.use('PDF')

from datetime import timedelta
from pytrade.algorithms.donchianchannels import DonchianTradingAlgorithm
from pytrade.algorithms.sma import SMATradingAlgorithm
import pytrade.algorithms.TAAnalysis as TAA
import pytrade.algorithms.MLAnalysis as MLA
from pytrade.backtesting.backtest import GoogleFinanceBacktest
from pytrade.feed import DynamicFeed
from pytrade.broker import PytradeBroker
from pytrade.base import TradingSystem
from pytrade.persistence.memprovider import MemoryDataProvider
from pytrade.persistence.sqliteprovider import SQLiteDataProvider
import pytradeapi
from pyalgotrade.tools import googlefinance
from pytrade.feed import DynamicFeed
from pyalgotrade.broker import Order
import pytz, datetime
from  pytrade.technicalindicator import TechnicalIndicator
import pytrade.technicalindicator.AD as AD
import pytrade.technicalindicator.SMA as SMA
import pytrade.technicalindicator.SupportResistence as SupportResistence


#codes = ["BBDC4","BDLL4","BGIP4","BOBR4","BRAP4","BRIV4","CMIG4","CRIV4","CTNM4","ELPL4","ESTR4","FJTA4","GETI4","GGBR4","GOAU4","GOLL4","GUAR4","INEP4","ITSA4","LAME4","LIXC4","MGEL4","MTSA4","MWET4","PCAR4","PETR4","POMO4","RAPT4","RCSL4","SAPR4","SHUL4","SLED4","TEKA4","TOYB4","TRPL4"]
codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3", "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3", "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4", "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3", "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
# codes = ["ABEV3", "BBAS3", "BBDC3"]

backtest = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2015, debugMode=False, csvStorage="./googlefinance")
#algorithm = DonchianTradingAlgorithm(backtest.getFeed(), backtest.getBroker(), 9, 26, 0.05)
#algorithm = SMATradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), longsize=15, shortsize=11, riskFactor=0.02)
#algorithm = SMATradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), longsize=50, shortsize=10, riskFactor=0.01)
#algorithm = MixTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), rsiPeriod=10, shortTrendSize=5, longTrendSize=20, supportLineSize=26, resistenceLineSize=9, riskFactor=0.05)
# algorithm = TAA.TAAnalysisTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), riskFactor=0.05, technicalIndicators={
#     TAA.AD_CODE:  TechnicalIndicator(backtest.getFeed(), TAA.AD_CODE, AD.ADOSCCalculator(size=20), newPlot=True),
#     TAA.LONGSMA_CODE: TechnicalIndicator(backtest.getFeed(), TAA.LONGSMA_CODE, SMA.SMACalculator(size=20), newPlot=False),
#     TAA.SHORTSMA_CODE: TechnicalIndicator(backtest.getFeed(), TAA.SHORTSMA_CODE, SMA.SMACalculator(size=5), newPlot=False),
#     TAA.SUPPORT_CODE: TechnicalIndicator(backtest.getFeed(), TAA.SUPPORT_CODE, SupportResistence.SuportCalculator(windowSize=26), newPlot=False),
#     TAA.RESISTENCE_CODE: TechnicalIndicator(backtest.getFeed(), TAA.RESISTENCE_CODE, SupportResistence.ResistenceCalculator(windowSize=9), newPlot=False)
# })
algorithm = MLA.MLAnalysisTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), riskFactor=0.05)
backtest.attachAlgorithm(algorithm)
backtest.run()

backtest.generateHtmlReport('/tmp/stock_analysis.html')
############################################################################################################################

# rowFilter = lambda row: row["Close"] == "-" or row["Open"] == "-" or row["High"] == "-" or row["Low"] == "-" or \
#                         row["Volume"] == "-"
# instruments = ["PETR4", "PETR3"]
# googleFeed = googlefinance.build_feed(codes, 2014, 2014, storage="./googlefinance", skipErrors=True,
#                                       rowFilter=rowFilter)
db = "./sqliteddb"
# feed = DynamicFeed    (db, codes, maxLen=10)
# feed.getDatabase().addBarsFromFeed(googleFeed)
################################################################################################
maxLen=int(26*1.4)
feed = DynamicFeed(db, codes, maxLen=maxLen)
days =  feed.getAllDays()

username="gabriel"
api = pytradeapi.PytradeApi(dbfilepah=db)
api.reinitializeUser(username=username, cash=10000)
tradingAlgorithmGenerator = lambda feed, broker: DonchianTradingAlgorithm(feed, broker, 9, 26, 0.05)

# utc = pytz.utc
# days = [
#             utc.localize(datetime.datetime(2014, 2, 7)),
#             utc.localize(datetime.datetime(2014, 2, 11))]


for i in range(len(days)):
    day = days[i]
    api = pytradeapi.PytradeApi(dbfilepah=db, username=username, tradingAlgorithmGenerator=tradingAlgorithmGenerator, codes=None, date=day, maxlen=maxLen, debugmode=False)
    api.executeAnalysis()
    api.persistData()

    if i == (len(days)-1):
        continue

    day = days[i+1]
    api = pytradeapi.PytradeApi(dbfilepah=db, username=username, tradingAlgorithmGenerator=tradingAlgorithmGenerator, codes=None, date=day, maxlen=maxLen,
                                debugmode=False)

    for order in api.getActiveMarketOrders()+api.getStopOrdersToConfirm():
        bar = api.getCurrentBarForInstrument(order.getInstrument())
        if bar is None:
            continue

        if not api.confirmOrder(order, bar.getDateTime(), order.getQuantity(), bar.getOpen(), 10):
            api.cancelOrder(order)

    api.persistData()
api.getEquity()



from pytradecli import PytradeCli
cli = PytradeCli(dbfilepah='./sqliteddb', maxlen=800)
cli.getAccountInfo()

cli.getApi().reinitializeUser(cash=10000, username='gabriel')

maxLen=int(26*1.4)
feed = DynamicFeed(db, codes, maxLen=maxLen)
allDays = feed.getAllDays()
utc = pytz.utc
specificdays = [
            utc.localize(datetime.datetime(2014, 2, 7)),
            utc.localize(datetime.datetime(2014, 2, 11)),
            utc.localize(datetime.datetime(2014, 9, 18)),
            utc.localize(datetime.datetime(2014, 10, 23)),
            utc.localize(datetime.datetime(2014, 10, 28)),
            utc.localize(datetime.datetime(2014, 12, 29))]
for i in range(len(allDays)):
    day = allDays[i]
    cli = PytradeCli(dbfilepah='./sqliteddb', date=day, maxlen=800)
    orders = cli.executeAnalysis()

    if i == (len(days) - 1):
        continue

    nextDay = allDays[allDays.index(day)+1]
    for order in orders:
        open = cli.getLastValuesForInstrument(order.getInstrument(), nextDay)[1]

        if not cli.confirmOrder(orderId=order.getId(), quantity=order.getQuantity(), price=open, commission=10, date=nextDay):
            cli.cancelOrder(order.getId())
    cli.save()
