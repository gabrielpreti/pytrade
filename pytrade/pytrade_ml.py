from pytrade.backtesting.backtest import GoogleFinanceBacktest
import pytrade.algorithms.MLAnalysis as MLA
import pandas as pd
import matplotlib.pyplot as plt
import os
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed


#############################################################################


def buy_result_function(open, close, high, low, volume, forecast_window_days):
    # return [all(map(lambda x: x>=0.9*v, close[i+1:i+forecast_window_days])) if i < len(close) - forecast_window_days else None for (i, v) in
    #                       list(enumerate(close))]
    return [
        close[i + forecast_window_days] > 1.1 * v if i < len(close) - forecast_window_days else None
        for (i, v)
        in list(enumerate(close))]


def sell_result_function(open, close, high, low, volume, forecast_window_days):
    # return [
    #     all(map(lambda x: x < v, data_set.close_price[i + 1:i + forecast_window_days])) if i < len(
    #         data_set.close_price) - forecast_window_days else None for (i, v) in
    #     list(enumerate(data_set.close_price))]
    return [
        close[i + forecast_window_days] < v if i < len(close) - forecast_window_days else None
        for (i, v)
        in list(enumerate(close))]


codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3",
         "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3",
         "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4", "LREN3",
         "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3", "SANB11",
         "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection2', SelectKBest(mutual_info_classif, k=100)),
    # ('classifier', DecisionTreeClassifier())
    # ('classifier', RandomForestClassifier(n_estimators=50, n_jobs=-1))
    ('classifier', SVC(kernel='rbf', C=1.0, random_state=0))
])
backtest = GoogleFinanceBacktest(instruments=codes[:10], initialCash=10000, fromYear=2014, toYear=2016, debugMode=False,
                                 csvStorage="./googlefinance")
algorithm = MLA.MLAnalysisTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), riskFactor=0.05,
                                           models=None, pipeline=pipeline, training_buy_window_days=400,
                                           forecast_buy_window_days=20, training_sell_window_days=400,
                                           forecast_sell_window_days=20, buy_result_function=buy_result_function,
                                           sell_result_function=sell_result_function)
backtest.attachAlgorithm(algorithm)
backtest.run()

backtest.generateHtmlReport('/tmp/stock_analysis.html')
plt.close('all')

####################################################
####################################################
buy_functions = {
    'BF1': lambda open, close, high, low, volume, forecast_window_days: [
        close[i + forecast_window_days] > v if i < len(close) - forecast_window_days else None
        for (i, v)
        in list(enumerate(close))],
    'BF2': lambda open, close, high, low, volume, forecast_window_days: [
        close[i + forecast_window_days] > 1.1 * v if i < len(close) - forecast_window_days else None
        for (i, v)
        in list(enumerate(close))],
    'BF3': lambda open, close, high, low, volume, forecast_window_days: [
        close[i + forecast_window_days] > 1.2 * v if i < len(close) - forecast_window_days else None
        for (i, v)
        in list(enumerate(close))],
    'BF4': lambda open, close, high, low, volume, forecast_window_days: [
        all(map(lambda x: x >= v, close[i + 1:i + forecast_window_days])) if i < len(
            close) - forecast_window_days else None for (i, v) in
        list(enumerate(close))],
    'BF5': lambda open, close, high, low, volume, forecast_window_days: [
        all(map(lambda x: x >= 0.95 * v, close[i + 1:i + forecast_window_days])) and close[
                                                                                         i + forecast_window_days] > v if i < len(
            close) - forecast_window_days else None for (i, v) in
        list(enumerate(close))],
    'BF6': lambda open, close, high, low, volume, forecast_window_days: [
        all(map(lambda x: x >= 0.9 * v, close[i + 1:i + forecast_window_days])) and close[
                                                                                        i + forecast_window_days] > v if i < len(
            close) - forecast_window_days else None for (i, v) in
        list(enumerate(close))]
}

sell_functions = {
    'SF1': lambda open, close, high, low, volume, forecast_window_days: [
        close[i + forecast_window_days] < v if i < len(close) - forecast_window_days else None
        for (i, v)
        in list(enumerate(close))],
    'SF2': lambda open, close, high, low, volume, forecast_window_days: [
        close[i + forecast_window_days] < 1.1 * v if i < len(close) - forecast_window_days else None
        for (i, v)
        in list(enumerate(close))],
    'SF3': lambda open, close, high, low, volume, forecast_window_days: [
        close[i + forecast_window_days] < 0.9 * v if i < len(close) - forecast_window_days else None
        for (i, v)
        in list(enumerate(close))],
    'SF4': lambda open, close, high, low, volume, forecast_window_days: [
        all(map(lambda x: x < v, close[i + 1:i + forecast_window_days])) if i < len(
            close) - forecast_window_days else None for (i, v) in
        list(enumerate(close))],
    'SF5': lambda open, close, high, low, volume, forecast_window_days: [
        all(map(lambda x: x < 0.95 * v, close[i + 1:i + forecast_window_days])) if i < len(
            close) - forecast_window_days else None for (i, v) in
        list(enumerate(close))],
    'SF6': lambda open, close, high, low, volume, forecast_window_days: [
        all(map(lambda x: x < 0.9 * v, close[i + 1:i + forecast_window_days])) if i < len(
            close) - forecast_window_days else None for (i, v) in
        list(enumerate(close))],
    'SF7': lambda open, close, high, low, volume, forecast_window_days: [
        any(map(lambda x: x < 0.9 * v, close[i + 1:i + forecast_window_days])) if i < len(
            close) - forecast_window_days else None for (i, v) in
        list(enumerate(close))],
    'SF8': lambda open, close, high, low, volume, forecast_window_days: [
        any(map(lambda x: x < 0.8 * v, close[i + 1:i + forecast_window_days])) if i < len(
            close) - forecast_window_days else None for (i, v) in
        list(enumerate(close))]
}

codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3",
         "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3",
         "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4", "LREN3",
         "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3", "SANB11",
         "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection2', SelectKBest(mutual_info_classif, k=100)),
    # ('classifier', DecisionTreeClassifier())
    # ('classifier', RandomForestClassifier(n_estimators=50, n_jobs=-1))
    ('classifier', SVC(kernel='rbf', C=1.0, random_state=0))
])


def run_algorithm(forecast_sell_window_days, buy_result_function, sell_result_function, fromYear=2015, toYear=2016, training_buy_window_days=400,
                  forecast_buy_window_days=20, training_sell_window_days=400, intermediary_results_file=None):
    print(
        "fromYear=%s toYear=%s, training_buy_window_days=%s, forecast_buy_window_days=%s, training_sell_window_days=%s, forecast_sell_window_days=%s" % (
            fromYear, toYear, training_buy_window_days, forecast_buy_window_days, training_sell_window_days,
            forecast_sell_window_days))

    if intermediary_results_file is not None and os.path.exists(intermediary_results_file):
        intermediary_results = pd.read_csv(intermediary_results_file, index_col=0)
        intermediary_results = intermediary_results[
            (intermediary_results.training_buy_window_days == training_buy_window_days) &
            (intermediary_results.forecast_buy_window_days == forecast_buy_window_days) &
            (intermediary_results.training_sell_window_days == training_sell_window_days) &
            (intermediary_results.forecast_sell_window_days == forecast_sell_window_days)
            ]
        if len(intermediary_results) > 0:
            result = intermediary_results.final_equity.values[0]
            print(
                "Found intermediary result for training_buy_window_days=%s, forecast_buy_window_days=%s, training_sell_window_days=%s, forecast_sell_window_days=%s with value %s. I'll not calculate it again" % (
                    training_buy_window_days, forecast_buy_window_days, training_sell_window_days,
                    forecast_sell_window_days, result))
            return result

    backtest = GoogleFinanceBacktest(instruments=codes, initialCash=10000, fromYear=fromYear, toYear=toYear,
                                     debugMode=False,
                                     csvStorage="./googlefinance")

    algorithm = MLA.MLAnalysisTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), riskFactor=0.05,
                                               models=None, pipeline=pipeline,
                                               training_buy_window_days=training_buy_window_days,
                                               forecast_buy_window_days=forecast_buy_window_days,
                                               training_sell_window_days=training_sell_window_days,
                                               forecast_sell_window_days=forecast_sell_window_days,
                                               buy_result_function=buy_result_function,
                                               sell_result_function=sell_result_function)
    backtest.attachAlgorithm(algorithm)
    backtest.run()
    return backtest.getBroker().getEquity()


results = pd.DataFrame(columns=('buy_function', 'sell_function', 'final_equity'))
for buy_function in buy_functions.keys():
    pool = Pool(processes=4)
    keys = sell_functions.keys()
    # sell_function_results = Parallel(n_jobs=-1,verbose=11)(delayed(run_algorithm(buy_result_function=buy_functions[buy_function], sell_result_function=sell_functions[i]))(i) for i in keys)
    sell_function_results = pool.map(lambda x: run_algorithm(buy_result_function=buy_functions[buy_function],
                                                             sell_result_function=sell_functions[x]), keys)
    for (i, k) in list(enumerate(keys)):
        results.loc[len(results)] = [buy_function, k, sell_function_results[i]]
    print(results)
# best result: BF1 and SF7
results.sort(columns='final_equity', inplace=True, ascending=False)
best_buy_function = results.head(1).buy_function.values[0]
best_sell_function = results.head(1).sell_function.values[0]

training_buy_window_days = [30, 60, 120, 200, 400, 600]
forecast_buy_window_days = [5, 10, 20, 40, 80, 150, 200]
training_sell_window_days = [30, 60, 120, 200, 400, 600]
forecast_sell_window_days = [5, 10, 20, 40, 80, 150, 200]
training_days_results = pd.DataFrame(columns=(
    'training_buy_window_days', 'forecast_buy_window_days', 'training_sell_window_days', 'forecast_sell_window_days',
    'final_equity'))
intermediary_results_file = "./intermediary_results.csv"
for tbwd in training_buy_window_days:
    for fbwd in forecast_buy_window_days:
        for tswd in training_sell_window_days:
            run_alg = partial(run_algorithm, buy_result_function=buy_functions['BF1'],
                              sell_result_function=sell_functions['SF7'],
                              fromYear=2014, toYear=2016, training_buy_window_days=tbwd,
                              forecast_buy_window_days=fbwd, training_sell_window_days=tswd,
                              intermediary_results_file=intermediary_results_file)
            pool = Pool(4)
            days_results = pool.map(run_alg, forecast_sell_window_days)

            if intermediary_results_file is not None and os.path.exists(intermediary_results_file):
                training_days_results = pd.read_csv(intermediary_results_file, index_col=0)
            for (i, k) in list(enumerate(forecast_sell_window_days)):
                training_days_results.loc[len(training_days_results)] = [tbwd, fbwd, tswd, k, days_results[i]]

            print(training_days_results)
            training_days_results.to_csv(intermediary_results_file)

# fromYear=2014 toYear=2016, training_buy_window_days=30, forecast_buy_window_days=5, training_sell_window_days=30, forecast_sell_window_days=80

#################################################
codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3",
         "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3",
         "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4",
         "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3", "SANB11",
         "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
codes = codes[:20]
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(mutual_info_classif, k='all')),
    # ('classifier', DecisionTreeClassifier())
    # ('classifier', RandomForestClassifier(n_estimators=50, n_jobs=-1))
    # ('classifier', SVC(kernel='linear', C=1.0, random_state=0))
    ('classifier', SVC(kernel='rbf', C=1.0, random_state=0))
    # ('classifier', MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5, 2)))
])
backtest = GoogleFinanceBacktest(instruments=codes, initialCash=1000000, fromYear=2015, toYear=2016, debugMode=False,
                                 csvStorage="./googlefinance")
algorithm = MLA.MLAnalysisTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), riskFactor=0.05,
                                           models=None, pipeline=pipeline, training_buy_window_days=200,
                                           forecast_buy_window_days=10, training_sell_window_days=30,
                                           forecast_sell_window_days=5, buy_result_function=buy_functions['BF1'],
                                           sell_result_function=sell_functions['SF7'])
backtest.attachAlgorithm(algorithm)
backtest.run()
plt.interactive(False)
backtest.generateHtmlReport('/tmp/report.html')
plt.close('all')
#1005462.25
#11 trades: 6 profitable, 5 unprofitable, 0 even
#$1007915.96
#9 trades: 4 profitable, 5 unprofitable, 0 even


trades = backtest.getTradesAnalyzer()

print("%s trades: %s profitable, %s unprofitable, %s even" %(trades.getCount(), trades.getProfitableCount(), trades.getUnprofitableCount(), trades.getEvenCount()))
