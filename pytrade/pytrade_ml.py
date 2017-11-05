import matplotlib
#matplotlib.use('PDF')
from sklearn.tree.tree import ExtraTreeClassifier

from pytrade.backtesting.backtest import GoogleFinanceBacktest
import numpy as np
from talib import SMA
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pandas.tools.plotting import scatter_matrix
from matplotlib.colors import ListedColormap
from talib import STOCHF
from talib import RSI
from talib import AD
from talib import DEMA
import talib

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression
import seaborn as sn
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.decomposition import PCA

codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3", "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3", "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4", "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3", "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]

def generate_data_set(open_values, close_values, high_values, low_values, volume_values, period):
    data_set = pd.DataFrame()
    data_set['open_price'] = open_values
    data_set['close_price'] = close_values
    data_set['high_price'] = high_values
    data_set['low_price'] = low_values
    data_set['volume_price'] = volume_values
    data_set['short_sma'] = SMA(close_values, 5)
    data_set['long_sma'] = SMA(close_values, 20)
    data_set['sma_diff'] = data_set.long_sma - data_set.short_sma
    data_set['stochf0'] = STOCHF(high=high_values, low=low_values, close=close_values)[0]
    data_set['stochf1'] = STOCHF(high=high_values, low=low_values, close=close_values)[1]
    data_set['rsi'] = RSI(close_values, 20)
    data_set['ad'] = AD(high=high_values, low=low_values, close=close_values, volume=volume_values)
    data_set['dema'] = DEMA(close_values)
    data_set['ema'] = talib.EMA(close_values)
    data_set['ht_trendiline'] = talib.HT_TRENDLINE(close_values)
    data_set['kama'] = talib.KAMA(close_values)
    data_set['midpoint'] = talib.MIDPOINT(close_values)
    data_set['midprice'] = talib.MIDPRICE(high=high_values, low=low_values)
    data_set['sar'] = talib.SAR(high=high_values, low=low_values)
    data_set['sarext'] = talib.SAREXT(high=high_values, low=low_values)
    data_set['adx'] = talib.ADX(high=high_values, low=low_values, close=close_values)
    data_set['adxr'] = talib.ADXR(high=high_values, low=low_values, close=close_values)
    data_set['apo'] = talib.APO(close_values)
    data_set['aroon0'] = talib.AROON(high=high_values, low=low_values)[0]
    data_set['aroon1'] = talib.AROON(high=high_values, low=low_values)[1]
    data_set['aroonosc'] = talib.AROONOSC(high=high_values, low=low_values)
    data_set['bop'] = talib.BOP(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['cmo'] = talib.CMO(close_values)
    data_set['dx'] = talib.DX(high=high_values, low=low_values, close=close_values)
    data_set['macdfix0'] = talib.MACDFIX(close_values)[0]
    data_set['macdfix1'] = talib.MACDFIX(close_values)[1]
    data_set['macdfix2'] = talib.MACDFIX(close_values)[2]
    data_set['mfi'] = talib.MFI(high=high_values, low=low_values, close=close_values, volume=volume_values)
    data_set['minus_di'] = talib.MINUS_DI(high=high_values, low=low_values, close=close_values)
    data_set['minus_dm'] = talib.MINUS_DM(high=high_values, low=low_values)
    data_set['mom'] = talib.MOM(close_values)
    data_set['plus_di'] = talib.PLUS_DI(high=high_values, low=low_values, close=close_values)
    data_set['plus_dm'] = talib.PLUS_DM(high=high_values, low=low_values)
    data_set['ppo'] = talib.PPO(close_values)
    data_set['roc'] = talib.ROC(close_values)
    data_set['stochf0'] = talib.STOCHF(high=high_values, low=low_values, close=close_values)[0]
    data_set['stochf1'] = talib.STOCHF(high=high_values, low=low_values, close=close_values)[1]
    data_set['stochrsi0'] = talib.STOCHRSI(close_values)[0]
    data_set['stochrsi1'] = talib.STOCHRSI(close_values)[1]
    # data_set['trix'] = talib.TRIX(close_values)
    data_set['ultosc'] = talib.ULTOSC(high=high_values, low=low_values, close=close_values)
    data_set['willr'] = talib.WILLR(high=high_values, low=low_values, close=close_values)
    data_set['adosc'] = talib.ADOSC(high=high_values, low=low_values, close=close_values, volume=volume_values)
    data_set['obv'] = talib.OBV(close_values, volume_values)
    data_set['ht_dcperiod'] = talib.HT_DCPERIOD(close_values)
    data_set['ht_dcphase'] = talib.HT_DCPHASE(close_values)
    data_set['ht_phasor0'] = talib.HT_PHASOR(close_values)[0]
    data_set['ht_phasor1'] = talib.HT_PHASOR(close_values)[1]
    data_set['ht_sine0'] = talib.HT_SINE(close_values)[0]
    data_set['ht_sine1'] = talib.HT_SINE(close_values)[1]
    data_set['ht_trendmode'] = talib.HT_TRENDMODE(close_values)
    data_set['atr'] = talib.ATR(high=high_values, low=low_values, close=close_values)
    data_set['trange'] = talib.TRANGE(high=high_values, low=low_values, close=close_values)

    data_set['CDL2CROWS'] = talib.CDL2CROWS(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDL3BLACKCROWS'] = talib.CDL3BLACKCROWS(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDL3INSIDE'] = talib.CDL3INSIDE(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDL3LINESTRIKE'] = talib.CDL3LINESTRIKE(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDL3OUTSIDE'] = talib.CDL3OUTSIDE(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDL3STARSINSOUTH'] = talib.CDL3STARSINSOUTH(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDL3WHITESOLDIERS'] = talib.CDL3WHITESOLDIERS(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLABANDONEDBABY'] = talib.CDLABANDONEDBABY(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLADVANCEBLOCK'] = talib.CDLADVANCEBLOCK(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLBELTHOLD'] = talib.CDLBELTHOLD(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLBREAKAWAY'] = talib.CDLBREAKAWAY(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLCLOSINGMARUBOZU'] = talib.CDLCLOSINGMARUBOZU(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLCONCEALBABYSWALL'] = talib.CDLCONCEALBABYSWALL(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLCOUNTERATTACK'] = talib.CDLCOUNTERATTACK(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLDARKCLOUDCOVER'] = talib.CDLDARKCLOUDCOVER(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLDOJI'] = talib.CDLDOJI(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLDOJISTAR'] = talib.CDLDOJISTAR(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLDRAGONFLYDOJI'] = talib.CDLDRAGONFLYDOJI(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLENGULFING'] = talib.CDLENGULFING(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLEVENINGDOJISTAR'] = talib.CDLEVENINGDOJISTAR(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLEVENINGSTAR'] = talib.CDLEVENINGSTAR(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLGAPSIDESIDEWHITE'] = talib.CDLGAPSIDESIDEWHITE(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLGRAVESTONEDOJI'] = talib.CDLGRAVESTONEDOJI(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLHAMMER'] = talib.CDLHAMMER(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLHANGINGMAN'] = talib.CDLHANGINGMAN(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLHARAMI'] = talib.CDLHARAMI(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLHARAMICROSS'] = talib.CDLHARAMICROSS(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLHIGHWAVE'] = talib.CDLHIGHWAVE(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLHIKKAKE'] = talib.CDLHIKKAKE(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLHIKKAKEMOD'] = talib.CDLHIKKAKEMOD(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLHOMINGPIGEON'] = talib.CDLHOMINGPIGEON(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLIDENTICAL3CROWS'] = talib.CDLIDENTICAL3CROWS(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLINNECK'] = talib.CDLINNECK(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLINVERTEDHAMMER'] = talib.CDLINVERTEDHAMMER(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLKICKING'] = talib.CDLKICKING(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLKICKINGBYLENGTH'] = talib.CDLKICKINGBYLENGTH(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLLADDERBOTTOM'] = talib.CDLLADDERBOTTOM(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLLONGLEGGEDDOJI'] = talib.CDLLONGLEGGEDDOJI(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLLONGLINE'] = talib.CDLLONGLINE(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLMARUBOZU'] = talib.CDLMARUBOZU(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLMATCHINGLOW'] = talib.CDLMATCHINGLOW(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLMATHOLD'] = talib.CDLMATHOLD(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLMORNINGDOJISTAR'] = talib.CDLMORNINGDOJISTAR(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLMORNINGSTAR'] = talib.CDLMORNINGSTAR(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLONNECK'] = talib.CDLONNECK(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLPIERCING'] = talib.CDLPIERCING(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLRICKSHAWMAN'] = talib.CDLRICKSHAWMAN(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLRISEFALL3METHODS'] = talib.CDLRISEFALL3METHODS(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLSEPARATINGLINES'] = talib.CDLSEPARATINGLINES(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLSHOOTINGSTAR'] = talib.CDLSHOOTINGSTAR(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLSHORTLINE'] = talib.CDLSHORTLINE(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLSPINNINGTOP'] = talib.CDLSPINNINGTOP(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLSTALLEDPATTERN'] = talib.CDLSTALLEDPATTERN(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLSTICKSANDWICH'] = talib.CDLSTICKSANDWICH(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLTAKURI'] = talib.CDLTAKURI(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLTASUKIGAP'] = talib.CDLTASUKIGAP(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLTHRUSTING'] = talib.CDLTHRUSTING(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLTRISTAR'] = talib.CDLTRISTAR(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLUNIQUE3RIVER'] = talib.CDLUNIQUE3RIVER(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLUPSIDEGAP2CROWS'] = talib.CDLUPSIDEGAP2CROWS(open=open_values, high=high_values, low=low_values, close=close_values)
    data_set['CDLXSIDEGAP3METHODS'] = talib.CDLXSIDEGAP3METHODS(open=open_values, high=high_values, low=low_values, close=close_values)

    data_set['result'] = [all(map(lambda x: x>v, close_values[i+1:i+period])) if i < len(close_values) - period else None for (i, v) in
                          list(enumerate(close_values))]
    # data_set['result'] = [open_values[i+period] > 1.01*v if i < len(close_values) - period else None for (i, v) in
    #                       list(enumerate(close_values))]
    return data_set.fillna(value=0)

pipe_clf = Pipeline([
    ('scaler', StandardScaler()),
    # ('feature_selection', VarianceThreshold()),
    ('feature_selection', SelectKBest(mutual_info_classif, k=5)),
    # ('feature_selection', PCA(n_components=10)),


    # ('clf', MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1))
    # ('classifier', SVC(kernel='rbf', C=1.0, random_state=0, probability=True))
    # ('classifier', DecisionTreeClassifier())
    # ('regressor', DecisionTreeRegressor())

    # ('clf', RandomForestClassifier(n_estimators=20))
    ('clf', LogisticRegression())
])

feed_2014 = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2014, debugMode=False,
                                     csvStorage="./googlefinance").getFeed()
feed_2014.loadAll()
feed_2015 = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2015, debugMode=False,
                                     csvStorage="./googlefinance").getFeed()
feed_2015.loadAll()
feed_2016 = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2016, debugMode=False,
                                     csvStorage="./googlefinance").getFeed()
feed_2016.loadAll()
sum_auc = 0
plt.rc('font', size=5)          # controls default text sizes
plt.rc('axes', titlesize=5)     # fontsize of the axes title
plt.rc('axes', labelsize=5)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=5)    # fontsize of the tick labels
plt.rc('ytick', labelsize=5)    # fontsize of the tick labels
plt.figure(1)
i=1
for code in codes:
    print("Analising %s" % (code))
    open_values_2014 = np.array(feed_2014.getDataSeries(instrument=code).getOpenDataSeries())
    close_values_2014 = np.array(feed_2014.getDataSeries(instrument=code).getCloseDataSeries())
    high_values_2014 = np.array(feed_2014.getDataSeries(instrument=code).getHighDataSeries())
    low_values_2014 = np.array(feed_2014.getDataSeries(instrument=code).getLowDataSeries())
    volume_values_2014 = np.array(feed_2014.getDataSeries(instrument=code).getVolumeDataSeries())
    data_set_2014 = generate_data_set(open_values_2014, close_values_2014, high_values_2014, low_values_2014, volume_values_2014, 10)

    X = data_set_2014.drop(labels=['result'], axis=1)
    y = [1 if y else 0 for y in data_set_2014.result]
    pipe_clf.fit(X, y)

    open_values_2015 = np.array(feed_2015.getDataSeries(instrument=code).getOpenDataSeries())
    close_values_2015 = np.array(feed_2015.getDataSeries(instrument=code).getCloseDataSeries())
    high_values_2015 = np.array(feed_2015.getDataSeries(instrument=code).getHighDataSeries())
    low_values_2015 = np.array(feed_2015.getDataSeries(instrument=code).getLowDataSeries())
    volume_values_2015 = np.array(feed_2015.getDataSeries(instrument=code).getVolumeDataSeries())
    data_set_2015 = generate_data_set(open_values_2015, close_values_2015, high_values_2015, low_values_2015, volume_values_2015, 10)

    X = data_set_2015.drop(labels=['result'], axis=1)
    y = [1 if y else 0 for y in data_set_2015.result]
    precision, recall, thresholds = metrics.precision_recall_curve(y[:], pipe_clf.predict_proba(X)[:, 1])
    auc = metrics.auc(recall, precision)
    if auc <=0.5:
        print("Skiping %s" % (code))
        continue

    X = data_set_2015.drop(labels=['result'], axis=1)
    y = [1 if y else 0 for y in data_set_2015.result]
    pipe_clf.fit(X, y)

    open_values_2016 = np.array(feed_2016.getDataSeries(instrument=code).getOpenDataSeries())
    close_values_2016 = np.array(feed_2016.getDataSeries(instrument=code).getCloseDataSeries())
    high_values_2016 = np.array(feed_2016.getDataSeries(instrument=code).getHighDataSeries())
    low_values_2016 = np.array(feed_2016.getDataSeries(instrument=code).getLowDataSeries())
    volume_values_2016 = np.array(feed_2016.getDataSeries(instrument=code).getVolumeDataSeries())
    data_set_2016 = generate_data_set(open_values_2016, close_values_2016, high_values_2016, low_values_2016, volume_values_2016, 10)

    X = data_set_2016.drop(labels=['result'], axis=1)
    y = [1 if y else 0 for y in data_set_2016.result]

    precision, recall, thresholds = metrics.precision_recall_curve(y[:], pipe_clf.predict_proba(X)[:,1])
    auc = metrics.auc(recall, precision)
    sum_auc += auc
    plt.subplot(6, 10, i)
    plt.plot(recall, precision)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('%s,AUC=%.2f' % (code, auc), fontdict={'fontsize':5})
    i += 1
plt.figure(1).suptitle("Sum AUC=%.2f" % (sum_auc))

#########################################################################################################################################
plt.scatter(data_set[data_set.result].short_sma, data_set[data_set.result].long_sma, color='green', label="positive")
plt.scatter(data_set[data_set.result==False].short_sma, data_set[data_set.result==False].long_sma, color='red', label="negative")
plt.xlabel("short_sma")
plt.ylabel("long_sma")
plt.legend(loc='best')

scatter_matrix(data_set)

def plot_decision_regions(X, y, classifier, resolution=0.02):
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)


#########################################################################################################################################
import pytrade.algorithms.MLAnalysis as MLA

codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3", "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3", "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4", "LREN3", "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3", "SANB11", "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
pipelines = {}
feed_2014 = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2014, debugMode=False,
                                     csvStorage="./googlefinance").getFeed()
feed_2014.loadAll()
feed_2015 = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2015, debugMode=False,
                                     csvStorage="./googlefinance").getFeed()
feed_2015.loadAll()
for code in codes:
    open_values = np.append(np.array(feed_2014.getDataSeries(instrument=code).getOpenDataSeries()),
                            np.array(feed_2015.getDataSeries(instrument=code).getOpenDataSeries()))
    close_values = np.append(np.array(feed_2014.getDataSeries(instrument=code).getCloseDataSeries()),
                             np.array(feed_2015.getDataSeries(instrument=code).getCloseDataSeries()))
    high_values = np.append(np.array(feed_2014.getDataSeries(instrument=code).getHighDataSeries()),
                            np.array(feed_2015.getDataSeries(instrument=code).getHighDataSeries()))
    low_values = np.append(np.array(feed_2014.getDataSeries(instrument=code).getLowDataSeries()),
                           np.array(feed_2015.getDataSeries(instrument=code).getLowDataSeries()))
    volume_values = np.append(np.array(feed_2014.getDataSeries(instrument=code).getVolumeDataSeries()),
                              np.array(feed_2015.getDataSeries(instrument=code).getVolumeDataSeries()))
    data_set = generate_data_set(open_values, close_values, high_values, low_values, volume_values, 30)

    X = data_set.drop(labels=['result'], axis=1)
    y = [1 if y else 0 for y in data_set.result]

    pipe_clf = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection2', SelectKBest(mutual_info_classif, k=30)),
        ('classifier', DecisionTreeClassifier())
    ])

    pipe_clf.fit(X, y)
    pipelines[code] = pipe_clf

backtest = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2016, debugMode=False, csvStorage="./googlefinance")
algorithm = MLA.MLAnalysisTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), riskFactor=0.05, models=pipelines)
backtest.attachAlgorithm(algorithm)
backtest.run()

backtest.generateHtmlReport('/tmp/stock_analysis.html')


#########################################################################################################################################
code = codes[0]
backtest = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2015, debugMode=False,
                                     csvStorage="./googlefinance")
backtest.getFeed().loadAll()
open_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getOpenDataSeries())
close_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getCloseDataSeries())
high_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getHighDataSeries())
low_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getLowDataSeries())
volume_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getVolumeDataSeries())
data_set = generate_data_set(open_values, close_values, high_values, low_values, volume_values, 15)
X = data_set.drop(labels=['result'], axis=1)
y = [1 if y else 0 for y in data_set.result]
rf = RandomForestClassifier(criterion='entropy')
rf.fit_transform(X, y)
sorted_features = sorted(zip(mutual_info_classif(X, y), X.columns), reverse=True)
plt.figure()
plt.title("Feature importances")
plt.bar(range(0, len(sorted_features)), [x[0] for x in sorted_features])
plt.xticks(range(0, len(sorted_features)), [x[1] for x in sorted_features], rotation='vertical')


#########################################################################################################################################
for code in codes:
    backtest = GoogleFinanceBacktest(instruments=codes, initialCash=10000, year=2015, debugMode=False,
                                         csvStorage="./googlefinance")
    backtest.getFeed().loadAll()
    open_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getOpenDataSeries())
    close_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getCloseDataSeries())
    high_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getHighDataSeries())
    low_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getLowDataSeries())
    volume_values = np.array(backtest.getFeed().getDataSeries(instrument=code).getVolumeDataSeries())
    data_set = generate_data_set(open_values, close_values, high_values, low_values, volume_values, 15)
    X = data_set.drop(labels=['result'], axis=1)
    y = [1 if y else 0 for y in data_set.result]

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import scale
    X = scale(X)

    n_pca_comp=3
    pca = PCA(n_components=n_pca_comp)
    Xtrans = pca.fit_transform(X, y)
    Xtrans = np.insert(Xtrans, n_pca_comp, y, axis=1)
    true_values = np.where(Xtrans[:, n_pca_comp]==1)
    false_values = np.where(Xtrans[:, n_pca_comp]==0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(code)
    ax.scatter(Xtrans[true_values, 0], Xtrans[true_values, 1], Xtrans[true_values, 2], c='blue')
    ax.scatter(Xtrans[false_values, 0], Xtrans[false_values, 1], Xtrans[false_values, 2], c='red')

#########################################################################################################################################
import pytrade.algorithms.MLAnalysis as MLA

pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('feature_selection2', SelectKBest(mutual_info_classif, k=10)),
        ('classifier', DecisionTreeClassifier())
    ])
backtest = GoogleFinanceBacktest(instruments=codes[:10], initialCash=10000, year=2016, debugMode=False, csvStorage="./googlefinance")
algorithm = MLA.MLAnalysisTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), riskFactor=0.05, models=None, pipeline=pipeline, training_window_days=100, forecast_window_days=30)
backtest.attachAlgorithm(algorithm)
backtest.run()

backtest.generateHtmlReport('/tmp/stock_analysis.html')
plt.close('all')
