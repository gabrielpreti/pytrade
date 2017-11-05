from pytrade.backtesting.backtest import GoogleFinanceBacktest
import numpy as np
import pandas as pd
import statsmodels.tsa.stattools as ts
import statsmodels.api as smapi
import pyalgotrade.technical.hurst as hurst


def get_halflife(s):
    s_lag = s.shift(1)
    s_lag.ix[0] = s_lag.ix[1]

    s_ret = s - s_lag
    s_ret.ix[0] = s_ret.ix[1]

    s_lag2 = smapi.add_constant(s_lag)

    model = smapi.OLS(s_ret,s_lag2)
    res = model.fit()

    halflife = round(-np.log(2) / res.params[1],0)
    return halflife

codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3", "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3", "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KROT3", "LAME4", "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
feed = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2015, debugMode=False,
                                     csvStorage="./googlefinance", filterInvalidRows=False).getFeed()
feed.loadAll()

for code in codes:
    hurstcoef = hurst.hurst_exp(feed.getDataSeries(instrument=code).getCloseDataSeries(), 2, 100)
    adf = ts.adfuller(feed.getDataSeries(instrument=code).getCloseDataSeries(), 1)
    halflife = get_halflife(pd.Series(list(feed.getDataSeries(instrument=code).getCloseDataSeries())))
    print("%s: hurstcoef=%s, adf=%s, halflive=%s" %(code, hurstcoef, adf[0]<adf[4]['10%'], halflife))

