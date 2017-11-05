from pytrade.backtesting.backtest import GoogleFinanceBacktest
import numpy as np
import pandas as pd
import pytrade.estimator as est

codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3", "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3", "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KROT3", "LAME4", "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
feed = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2015, debugMode=False,
                                     csvStorage="./googlefinance", filterInvalidRows=False).getFeed()
feed.loadAll()

moviments_frame = pd.DataFrame()
for code in codes:
    closevalues = np.array(feed.getDataSeries(instrument=code).getCloseDataSeries())[-15:]
    moviments_frame[code] = [(closevalues[i]-closevalues[i-1])/closevalues[i-1] if i>0 else 0 for i in range(len(closevalues))]

np.corrcoef(moviments_frame, rowvar=False)
np.corrcoef(moviments_frame[codes[0]], moviments_frame[codes[1]])[0, 1]

shift = 1
threshold = 0.7
for code in codes:
    closevalues = np.roll(moviments_frame[code], shift=-shift)
    closevalues = closevalues[:-shift]
    toppers = {}
    for c in list(set(codes) - set([code])):
        corrcoef = np.corrcoef(closevalues, np.array(moviments_frame[c])[:-shift])[0, 1]
        if abs(corrcoef)>= threshold:
            toppers[c] = corrcoef
    print("###########\n %s toppers for %s: \n %s" % (len(toppers), code, toppers))


###########################################################################
import pytrade.algorithms.MLAnalysis as MLA
import matplotlib.pyplot as plt


codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3", "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3", "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KROT3", "LAME4", "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
feed = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2014, debugMode=False, csvStorage="./googlefinance", filterInvalidRows=False).getFeed()
feed.loadAll()
moviments_frame = pd.DataFrame()
for code in codes:
    closevalues = np.array(feed.getDataSeries(instrument=code).getCloseDataSeries())
    moviments_frame[code] = closevalues

estimator = est.StochCorrelationEstimator(threshold=0.7, shift=1, histsize=15)
estimator.fit(moviments_frame)

backtest = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2015, debugMode=False, csvStorage="./googlefinance")
algorithm = MLA.CorrelationAnalysisTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), riskFactor=0.05, model=estimator)
backtest.attachAlgorithm(algorithm)
backtest.run()

backtest.generateHtmlReport('/tmp/stock_analysis.html')
plt.close("all")
