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

    @property
    def _state(self):
        return (self._regressors, self.features, self.outputs)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._state, f)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)

        return cls(*state)

    @classmethod
    def from_features(cls, features=FEATURES, outputs=OUTPUTS):
        regressors = {
            output: BayesianRidge(compute_score=True)
            for output in outputs
        }

        return cls(regressors, features, outputs)

    def predict(self, X):
        return {
            output: self.predict_output(X[output], output)
            for output, reg in self._regressors.items()
        }

    def predict_output(self, X, output):
        return self._regressors[output].predict(X)

    def predict_df(self, df):
        #TODO: finish implementing this method
        for out in self.outputs:
            idx = df[self.features + [out]].dropna().index
            X = df.loc[idx, self.features].values
            y = df.loc[idx, out].values.reshape(-1)
            self.fit_output(X, y, out)

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

    def get_stats(self, X, y, output):
        y_pred = self.predict_output(X, output)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        rsq, _ = stats.pearsonr(y, y_pred)
        
        return rmse, rsq

