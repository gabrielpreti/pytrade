from pytrade.algorithms.donchianchannels import DonchianTradingAlgorithm
from pytradeapi import PytradeApi
from texttable import Texttable
from datetime import datetime
from pyalgotrade.broker import Order


#CARREGAR COTACOES          DONE
#INFORMACOES CONTA          DONE
    #SALDO
    #ORDENS EM ABERTO
    #SHARES
    #EQUITY
#Gerar ordens (executar analise)
#Confirmar ordem (gerar share)
    #Data
    #Valor
    #Quantidade

class PytradeCli(object):
    def __init__(self, dbfilepah="/var/pytrade/sqlitedb", googleFinanceDir="/var/pytrade/googlefinance", date=datetime.now(), maxlen=90, codes=None, tradingAlgorithmGenerator=None):
        if codes is None:
            codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3",
             "CCRO3", "CIEL3", "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3",
             "EGIE3", "EMBR3", "ENBR3", "EQTL3", "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4",
             "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4", "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3",
             "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3", "SANB11", "SBSP3", "SMLE3",
             "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]

        if tradingAlgorithmGenerator is None:
            tradingAlgorithmGenerator = lambda feed, broker: DonchianTradingAlgorithm(feed, broker, 9, 26, 0.05)

        self.__date = date
        self.__username = 'gabriel'
        self.__api = PytradeApi(dbfilepah=dbfilepah, googleFinanceDir=googleFinanceDir, username=self.__username, tradingAlgorithmGenerator=tradingAlgorithmGenerator, codes=codes, date=self.__date, maxlen=maxlen, debugmode=False)

    def getApi(self):
        return self.__api

    def updateStockData(self):
        print "Updating stock data ..."
        self.__api.updateStockData()

        print "Stock data updated until %s" % self.__api.getLastStockDate()

    def getLastStockDate(self):
        return self.__api.getLastStockDate()

    def getAccountInfo(self):
        cash = self.__api.getCash()
        equity = self.__api.getEquity()
        stopOrders = self.__api.getStopOrders()
        marketOrders = self.__api.getActiveMarketOrders()
        shares = self.__api.getAllShares()

        print "Available cash: R${:,.2f} \t\t\t Equity: R${:,.2f}".format(cash, equity)
        self.printStopOrders(stopOrders if stopOrders is not None else [])
        self.printMarketOrders(marketOrders if marketOrders is not None else [])
        self.printShares(shares if shares is not None else {})

        return (cash, equity, stopOrders, marketOrders, shares)

    def printShares(self, shares):
        print "\nShares:"
        sharesTable = Texttable(max_width=0)
        sharesTable.header(["Stock Code", "Quantity", "Estimated Value (Date)"])
        for code, quantity in shares.items():
            currentBars = self.__api.getCurrentBarForInstrument(code)

            sharesTable.add_row([
                code,  # code
                "{:,.2f}".format(quantity),  # quantity
                "R${:,.2f} ({:%d/%b/%Y})".format(currentBars.getClose() * quantity, currentBars.getDateTime())
                # Estimated Value
            ])
        print sharesTable.draw()

    def printMarketOrders(self, marketOrders):
        print "\nMarket Orders:"
        marketOrdersTable = Texttable(max_width=0)
        marketOrdersTable.header(['Id', 'Action', 'Stock Code', 'Quantity', 'State', 'Creation Date (Close Value)',
                                  'Total Initial Estimated Value', 'Stop Loss Value', 'Current Value',
                                  'Total Current Estimated Value'])
        for o in marketOrders:
            code = o.getInstrument()
            quantity = o.getQuantity()
            stockValues = self.__api.getLastValuesForInstrument(code, o.getSubmitDateTime())
            currentBars = self.__api.getCurrentBarForInstrument(code)

            marketOrdersTable.add_row([
                o.getId(),  # id
                Order.Action.toString(Order.Action.fromInteger(o.getAction())),  # action
                code,  # code
                "{:,.2f}".format(o.getQuantity()),  # quantity
                Order.State.toString(o.getState()),  # state
                "{:%d/%b/%Y} (R${:,.2f})".format(o.getSubmitDateTime(), stockValues[4]),  # creation date
                "R${:,.2f}".format(quantity * stockValues[4]),  # total initial estimated value
                "R${:,.2f}".format(o.stopLossValue) if o.getAction() == Order.Action.BUY else "-",  # stop loss value
                "R${:,.2f} ({:%d/%b/%Y})".format(currentBars.getClose(), currentBars.getDateTime()),  # current value
                "R${:,.2f}".format(currentBars.getClose() * quantity)  # total current estimanted value
            ]

            )
        print marketOrdersTable.draw()

    def printStopOrders(self, stopOrders):
        print "\nStop Orders:"
        stopOrdersTable = Texttable(max_width=0)
        stopOrdersTable.header(
            ['Id', 'Stock Code', 'Quantity', 'Creation Date', 'Stop Value', 'State', 'Current Value'])
        for o in stopOrders:
            code = o.getInstrument()
            currentBars = self.__api.getCurrentBarForInstrument(code)

            stopOrdersTable.add_row([
                o.getId(),  # id
                code,  # code
                "{:,.2f}".format(o.getQuantity()),  # quantity
                o.getSubmitDateTime().strftime("%d/%b/%Y"),  # creation date
                "R${:,.2f}".format(o.getStopPrice()),  # stop price
                Order.State.toString(o.getState()),  # state
                "R${:,.2f} ({:%d/%b/%Y})".format(currentBars.getClose(), currentBars.getDateTime())  # current value
            ])
        print stopOrdersTable.draw()

    def executeAnalysis(self):
        self.__api.executeAnalysis()

        marketOrders = self.__api.getActiveMarketOrders()
        stopOrders = self.__api.getStopOrdersToConfirm()

        self.printStopOrders(stopOrders)
        self.printMarketOrders(marketOrders)

        return marketOrders + stopOrders

    def confirmOrder(self, orderId, quantity, price, commission, date=datetime.now()):
        ret = self.__api.confirmOrder(
            order=self.__api.getOrderById(orderId),
            datetime=date,
            quantity=quantity,
            price=price,
            commission=commission
        )
        print "Order confirmed" if ret else "Order not confirmed"
        return ret

    def cancelOrder(self, orderId):
        self.__api.cancelOrder(self.__api.getOrderById(orderId))

    def save(self):
        self.__api.persistData()

    def getLastValuesForInstrument(self, instrument, date):
        values = self.__api.getLastValuesForInstrument(instrument, date)
        table = Texttable(max_width=0)
        table.header(["Date", "Open", "High", "Low", "Close", "Volume"])
        table.add_row([
            "{:%d/%b/%Y}".format(values[0]),
            "R${:,.2f}".format(values[1]),
            "R${:,.2f}".format(values[2]),
            "R${:,.2f}".format(values[3]),
            "R${:,.2f}".format(values[4]),
            "R${:,.2f}".format(values[5]),
        ])
        print(table.draw())
        return values


