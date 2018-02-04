import matplotlib

matplotlib.use('pdf')

from pytrade.backtesting.backtest import GoogleFinanceBacktest
import pytrade.algorithms.MLAnalysis as MLA
import pandas as pd
import matplotlib.pyplot as plt
import os
from functools import partial
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest

import multiprocessing
from pathos.multiprocessing import ProcessingPool as Pool
from joblib import Parallel, delayed
import itertools

plt.interactive(False)

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
codes = codes[:20]
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection2', SelectKBest(mutual_info_classif, k=100)),
    # ('classifier', DecisionTreeClassifier())
    ('classifier', RandomForestClassifier(n_estimators=50, n_jobs=-1))
    # ('classifier', SVC(kernel='rbf', C=1.0, random_state=0))
])

# results = pd.DataFrame(columns=('buy_function', 'sell_function', 'final_equity'))
# for buy_function in buy_functions.keys():
#     pool = Pool(processes=4)
#     keys = sell_functions.keys()
#     # sell_function_results = Parallel(n_jobs=-1,verbose=11)(delayed(run_algorithm(buy_result_function=buy_functions[buy_function], sell_result_function=sell_functions[i]))(i) for i in keys)
#     sell_function_results = pool.map(lambda x: run_algorithm(buy_result_function=buy_functions[buy_function],
#                                                              sell_result_function=sell_functions[x]), keys)
#     for (i, k) in list(enumerate(keys)):
#         results.loc[len(results)] = [buy_function, k, sell_function_results[i]]
#     print(results)
# # best result: BF1 and SF7
# results.sort(columns='final_equity', inplace=True, ascending=False)
# best_buy_function = results.head(1).buy_function.values[0]
# best_sell_function = results.head(1).sell_function.values[0]

LOCK = multiprocessing.Lock()


def run_algorithm(buy_result_function, sell_result_function, fromYear=2015, toYear=2016,
                  training_buy_window_days=400,
                  forecast_buy_window_days=20, training_sell_window_days=400, forecast_sell_window_days=20,
                  results_directory=None, intermediary_results_file=None):
    print(
        "fromYear=%s toYear=%s, training_buy_window_days=%s, forecast_buy_window_days=%s, training_sell_window_days=%s, forecast_sell_window_days=%s, intermediary_results_file=%s" % (
            fromYear, toYear, training_buy_window_days, forecast_buy_window_days, training_sell_window_days,
            forecast_sell_window_days, intermediary_results_file))

    intermediary_results_file = "%s/%s" % (results_directory, intermediary_results_file)
    if os.path.exists(intermediary_results_file):
        LOCK.acquire()
        intermediary_results = pd.read_csv(intermediary_results_file, index_col=0)
        LOCK.release()
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

    initial_cash = 1000000
    backtest = GoogleFinanceBacktest(instruments=codes, initialCash=initial_cash, fromYear=fromYear, toYear=toYear,
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
    equity = backtest.getBroker().getEquity()
    trades = backtest.getTradesAnalyzer()


    LOCK.acquire()
    intermediary_results = pd.read_csv(intermediary_results_file, index_col=0) if os.path.exists(
        intermediary_results_file) else pd.DataFrame(columns=(
        'training_buy_window_days', 'forecast_buy_window_days', 'training_sell_window_days',
        'forecast_sell_window_days', 'initial_equity', 'final_equity', 'total_trades', 'profitable_trades',
        'unprofitable_trades', 'even_trades'))
    intermediary_results = intermediary_results.reindex(range(len(intermediary_results)))
    intermediary_results.loc[len(intermediary_results)] = [training_buy_window_days, forecast_buy_window_days,
                                                           training_sell_window_days, forecast_sell_window_days,
                                                           initial_cash, equity, trades.getCount(),
                                                           trades.getProfitableCount(),
                                                           trades.getUnprofitableCount(), trades.getEvenCount()]
    intermediary_results.to_csv(intermediary_results_file)
    backtest.generateHtmlReport("%s/%s_%s_%s_%s.html" % (results_directory,
                                                         training_buy_window_days, forecast_buy_window_days,
                                                         training_sell_window_days, forecast_sell_window_days))
    # plt.close('all')
    LOCK.release()

    return equity


n_cores = multiprocessing.cpu_count() - 1
pool = Pool(n_cores)
# training_buy_window_days = [30, 60, 120, 200, 400, 600]
training_buy_window_days = [120, 200]
forecast_buy_window_days = [5, 10, 20, 40, 80, 150, 200]
training_sell_window_days = [30, 60, 120, 200, 400, 600]
# forecast_sell_window_days = [5, 10, 20, 40, 80, 150, 200]
forecast_sell_window_days = [5, 10]

days_product = zip(*list(
    itertools.product(training_buy_window_days, forecast_buy_window_days, training_sell_window_days,
                      forecast_sell_window_days)))
comb_num = len(days_product[0])
buy_result_function = [buy_functions['BF1']] * comb_num
sell_result_function = [sell_functions['SF7']] * comb_num
fromYear = [2015] * comb_num
toYear = [2016] * comb_num
training_buy_window_days = days_product[0]
forecast_buy_window_days = days_product[1]
training_sell_window_days = days_product[2]
forecast_sell_window_days = days_product[3]
# intermediary_results_file = ["./intermediary_results.csv"]*comb_num
results_directory = ["/export/pytrade/"] * comb_num
intermediary_results_file = ["intermediary_results.csv"] * comb_num
pool.map(run_algorithm, buy_result_function, sell_result_function, fromYear, toYear, training_buy_window_days,
         forecast_buy_window_days, training_sell_window_days, forecast_sell_window_days, results_directory,
         intermediary_results_file)
