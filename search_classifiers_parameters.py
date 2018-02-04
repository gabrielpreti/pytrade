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
import itertools


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

def run_algorithm(scaller, selector, classifier):
    pipeline = Pipeline([
        ('scaler', scaller),
        ('feature_selection', selector),
        ('classifier', classifier)
    ])
    backtest = GoogleFinanceBacktest(instruments=codes, initialCash=1000000, fromYear=2015, toYear=2016,
                                     debugMode=False,
                                     csvStorage="./googlefinance")
    algorithm = MLA.MLAnalysisTradingAlgorithm(feed=backtest.getFeed(), broker=backtest.getBroker(), riskFactor=0.05,
                                               models=None, pipeline=pipeline, training_buy_window_days=200,
                                               forecast_buy_window_days=10, training_sell_window_days=30,
                                               forecast_sell_window_days=5, buy_result_function=buy_functions['BF1'],
                                               sell_result_function=sell_functions['SF7'])
    backtest.attachAlgorithm(algorithm)
    backtest.run()
    return backtest.getBroker().getEquity()


scallers = [StandardScaler()]
selectors = [SelectKBest(mutual_info_classif, k=10), SelectKBest(mutual_info_classif, k=30), SelectKBest(mutual_info_classif, k=50), SelectKBest(mutual_info_classif, k=100), SelectKBest(mutual_info_classif, k=200), SelectKBest(mutual_info_classif, k='all') ]
# classifiers = [SVC(kernel='rbf', C=1.0, random_state=0), SVC(kernel='linear', C=1.0, random_state=0),SVC(kernel='poly', C=1.0, random_state=0),SVC(kernel='sigmoid', C=1.0, random_state=0),DecisionTreeClassifier(),RandomForestClassifier(n_estimators=10, n_jobs=1), RandomForestClassifier(n_estimators=20, n_jobs=1), RandomForestClassifier(n_estimators=50, n_jobs=1), GaussianNB(), MLPClassifier(), MLPClassifier(hidden_layer_sizes=(5, 2)), MLPClassifier(hidden_layer_sizes=(10, 5)), MLPClassifier(hidden_layer_sizes=(50, 20)), MLPClassifier(solver='lbfgs'), MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5, 2)), MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 5)), MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50, 20))]
classifiers = [RandomForestClassifier(n_estimators=20, n_jobs=1), RandomForestClassifier(n_estimators=50, n_jobs=1), GaussianNB(), MLPClassifier(), MLPClassifier(hidden_layer_sizes=(5, 2)), MLPClassifier(hidden_layer_sizes=(10, 5)), MLPClassifier(hidden_layer_sizes=(50, 20)), MLPClassifier(solver='lbfgs'), MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5, 2)), MLPClassifier(solver='lbfgs', hidden_layer_sizes=(10, 5)), MLPClassifier(solver='lbfgs', hidden_layer_sizes=(50, 20))]
meta_parameters = zip(*list(
    itertools.product(scallers, selectors, classifiers)))

codes = ["ABEV3", "BBAS3", "BBDC3", "BBDC4", "BBSE3", "BRAP4", "BRFS3", "BRKM5", "BRML3", "BVMF3", "CCRO3", "CIEL3",
         "CMIG4", "CPFE3", "CPLE6", "CSAN3", "CSNA3", "CTIP3", "CYRE3", "ECOR3", "EGIE3", "EMBR3", "ENBR3", "EQTL3",
         "ESTC3", "FIBR3", "GGBR4", "GOAU4", "HYPE3", "ITSA4", "ITUB4", "JBSS3", "KLBN11", "KROT3", "LAME4",
         "MRFG3", "MRVE3", "MULT3", "NATU3", "PCAR4", "PETR3", "PETR4", "QUAL3", "RADL3", "RENT3", "RUMO3", "SANB11",
         "SBSP3", "SMLE3", "SUZB5", "TIMP3", "UGPA3", "USIM5", "VALE3", "VALE5", "VIVT4", "WEGE3"]
codes = codes[:20]

n_cores = multiprocessing.cpu_count()
pool = Pool(n_cores)
results = pool.map(run_algorithm, meta_parameters[0], meta_parameters[1], meta_parameters[2])
result_tuples = []
for i in range(0, len(results)):
    result_tuples.append((results[i], meta_parameters[0][i], meta_parameters[1][i], meta_parameters[2][i]))
sorted(result_tuples, key=lambda r: r[0])

with open('/home/gsantiago/search_classifiers.txt', mode='w') as f:
    f.write(str(sorted(result_tuples, key=lambda r: r[0])))




##BEST#####################################
#(1007915.9600000002,
# StandardScaler(copy=True, with_mean=True, with_std=True),
# SelectKBest(k='all',
#       score_func=<function mutual_info_classif at 0x7f779e755ed8>),
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#   max_iter=-1, probability=False, random_state=0, shrinking=True,
#   tol=0.001, verbose=False))

# (1007040.9799999995,
#  StandardScaler(copy=True, with_mean=True, with_std=True),
#  SelectKBest(k=100,
#              score_func= < function
# mutual_info_classif
# at
# 0x7f15faebfed8 >),
# SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
#     decision_function_shape=None, degree=3, gamma='auto', kernel='sigmoid',
#     max_iter=-1, probability=False, random_state=0, shrinking=True,
#     tol=0.001, verbose=False))



