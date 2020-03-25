import pickle
import pandas as pd
import numpy as np
from scipy import stats

from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def LatticePredictor:
    def __init__(self, regressor):
        self._regressor = regressor

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self._regressor, f)

    @classmethod
    def from_file(cls, filename):
        with open(filename, 'rb') as f:
            regressor = pickle.load(self, f)

        return cls(regressor)

    def predict(self, X):
        return self._regressor.predict(X)

    def get_stats(self, X, y):
        y_pred = self.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        rsq, _ = stats.pearsonr(y, y_pred)
        
        return rmse, rsq

