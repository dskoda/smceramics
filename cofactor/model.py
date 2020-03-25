import pickle
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


FEATURES = ['T', 'en_p', 'ea', 'valence', 'rad_slater', 'rad_clementi']
OUTPUTS = ['tetr_a', 'tetr_c', 'mono_a', 'mono_b', 'mono_c', 'mono_beta']


class LatticePredictor:
    def __init__(
        self,
        regressors,
        features,
        outputs
    ):
        self._regressors = regressors
        self.features = features
        self.outputs = outputs

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._regressors, f)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as f:
            regressors = pickle.load(f)

        return cls(regressors)

    @classmethod
    def from_features(cls, features=FEATURES, outputs=OUTPUTS):
        regressors = {
            output: BayesianRidge(compute_score=True)
            for output in outputs
        }

        return cls(regressors, features, outputs)

    def predict(self, X):
        return {
            output: reg.predict(X)
            for output, reg in self._regressors.items()
        }

    def fit(self, X, y):
        """X and y are hashables with the same keys of
            self.outputs (and regressors)
        """
        for output, reg in self._regressors.items():
            self.fit_output(X[output], y[output], output)

    def fit_output(self, X, y, output):
        self._regressors[output].fit(X, y)

    def fit_df(self, df):
        for out in self.outputs:
            idx = df[self.features + [out]].dropna().index
            X = df.loc[idx, self.features].values
            y = df.loc[idx, out].values.reshape(-1)
            self.fit_output(X, y, out)

    def get_stats(self, X, y):
        y_pred = self.predict(X)
        rmse = {
            out: np.sqrt(mean_squared_error(y, pred))
            for out, pred in y_pred.items()
        }
        rsq = {
            out: stats.pearsonr(y, pred)[0]
            for out, pred in y_pred.items()
        }
        
        return rmse, rsq

