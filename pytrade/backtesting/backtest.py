from pyalgotrade import logger
from pyalgotrade.barfeed import googlefeed
from pyalgotrade.tools import googlefinance
from pyalgotrade.broker import backtesting
from pytrade.base import TradingSystem
from pyalgotrade.stratanalyzer import returns
from pytrade.backtesting.analyzer.dalytradingresults import DailyTradingResults
from pyalgotrade import plotter
from pyalgotrade.plotter import SecondaryMarker
import mpld3
from mpld3 import plugins

class GoogleFinanceBacktest(object):

    LOGGER_NAME = "GoogleFinanceBacktest"

    def __init__(self, instruments, initialCash, year, debugMode=True, csvStorage="./googlefinance"):
        self.__logger = logger.getLogger(GoogleFinanceBacktest.LOGGER_NAME)
        self.__finalPortfolioValue = 0

        # Create Feed
        self.__feed = googlefeed.Feed()
        rowFilter = lambda row: row["Close"] == "-" or row["Open"] == "-" or row["High"] == "-" or row["Low"] == "-" or \
                                row["Volume"] == "-"

        self.__feed = googlefinance.build_feed(instruments, year, year, storage=csvStorage, skipErrors=True, rowFilter=rowFilter)

        # Create Broker
        comissionModel = backtesting.FixedPerTrade(10)
        self.__broker = backtesting.Broker(initialCash, self.__feed, commission=comissionModel)
        self.__strategy = TradingSystem(self.__feed, self.__broker, debugMode=debugMode)

        # Create Analyzers
        returnsAnalyzer = returns.Returns()
        self.__strategy.attachAnalyzer(returnsAnalyzer)
        dailyResultsAnalyzer = DailyTradingResults()
        self.__strategy.attachAnalyzer(dailyResultsAnalyzer)

        # Create plotters
        self.__plotters = []
        self.__plotters.append(
            plotter.StrategyPlotter(self.__strategy, plotAllInstruments=False, plotPortfolio=True, plotBuySell=False))
        self.__plotters[0].getOrCreateSubplot("returns").addDataSeries("Simple returns", returnsAnalyzer.getReturns())
        self.__plotters[0].getOrCreateSubplot("dailyresult").addDataSeries("Daily Results", dailyResultsAnalyzer.getTradeResults())

        for i in range(0, len(instruments)):
            p = plotter.StrategyPlotter(self.__strategy, plotAllInstruments=False, plotPortfolio=False)
            p.getInstrumentSubplot(instruments[i])
            self.__plotters.append(p)

    def getBroker(self):
        return self.__broker

    def getFeed(self):
        return self.__feed

    def attachAlgorithm(self, tradingAlgorithm):
        self.__strategy.setAlgorithm(tradingAlgorithm)

    def run(self):
        self.__strategy.run()
        self.__finalPortfolioValue = self.__strategy.getBroker().getEquity()
        self.__logger.info("Final portfolio value: $%.2f" % self.__strategy.getBroker().getEquity())

    def generateHtmlReport(self, htmlfilepath):
        figures = []
        htmlcontent = ""
        for p in self.__plotters:
            subplots = p.getSubplots()
            for instrument, subplot in subplots.items():
                for ti in self.__strategy.getAlgorithm().getTechnicalIndicators().values():
                    if ti.isNewPlot():
                        p.getOrCreateSubplot(instrument + " " + ti.getName()).addAndProcessDataSeries(instrument + " " + ti.getName(), ti[instrument])
                    else:
                        subplot.addAndProcessDataSeries(instrument + " " + ti.getName(), ti[instrument], defaultClass=SecondaryMarker)
            fig = p.buildFigure()
            figures.append(fig)
            htmlcontent += self.__generatehtml(fig)

        with open(htmlfilepath, mode="w") as htmlfile:
            htmlfile.write(htmlcontent)
        return figures

    def __generatehtml(self, fig):
        i = 0
        for ax in fig.get_axes():
            plugins.connect(fig, plugins.InteractiveLegendPlugin(plot_elements=ax.get_lines(),
                                                                 labels=[str(x) for x in
                                                                         ax.get_legend().get_texts()], ax=ax,
                                                                 alpha_unsel=0.0, alpha_over=1.5))
            i += 1
            for line in ax.get_lines():
                line.set_ydata([x if x is not None else 0 for x in line.get_ydata()])
                plugins.connect(fig, plugins.PointLabelTooltip(points=line,
                                                               labels=[str(y) for y in line.get_ydata()],
                                                               hoffset=10, voffset=10))

        return mpld3.fig_to_html(fig)

