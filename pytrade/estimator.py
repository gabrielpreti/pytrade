from sklearn.base import BaseEstimator, RegressorMixin
import pandas as pd
import numpy as np

class StochCorrelationEstimator(BaseEstimator, RegressorMixin):

    def __init__(self, shift=1, threshold=0.8, evalcodeparam="evalcode", histsize=None):
        self.shift = shift
        self.threshold = threshold
        self.evalcodeparam = evalcodeparam
        self.histsize = histsize

    def fit(self, X, y=None):
        assert (type(self.shift) == int), "shift parameter must be integer"
        assert (type(self.threshold) == float), "threshold parameter must be float"
        assert (type(self.evalcodeparam) == str), "evalcodeparam parameter must be str"
        assert (type(self.histsize) == int), "histsize parameter must be int"

        moviments_frame = pd.DataFrame()
        stoch_codes = X.columns.values
        for code in stoch_codes:
            values = np.array(X[code])
            if self.histsize is not None:
                values = values[-self.histsize:]
            moviments_frame[code] = [(values[i] - values[i - 1]) / values[i - 1] if i > 0 else 0 for i in range(len(values))]

        self.model_ = {}
        for code in stoch_codes:
            values = np.roll(moviments_frame[code], shift=-self.shift)
            values = values[:-self.shift]
            toppers = {}
            for c in list(set(stoch_codes) - set([code])):
                corrcoef = np.corrcoef(values, np.array(moviments_frame[c])[:-self.shift])[0, 1]
                if abs(corrcoef) >= self.threshold:
                    toppers[c] = corrcoef
            self.model_[code] = toppers

        return self

    def predict(self, X):
        output = []
        for i, row in X.iterrows():
            evalcode = row[self.evalcodeparam]
            score = 0;
            model = self.model_[evalcode]
            for k in model.keys():
                score += model[k]*row[k]
            output.append(score)

        return output


