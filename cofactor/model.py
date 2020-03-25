import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import itertools
import tqdm

from sklearn.linear_model import LinearRegression, BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def get_stats(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rsq, _ = stats.pearsonr(y_true, y_pred)
    
    return rmse, rsq

reg  = BayesianRidge(compute_score=True)



