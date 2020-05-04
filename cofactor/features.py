import pandas as pd
import numpy as np
import itertools


class Features:
    def __init__(self, atomic_feat):
        self.atomic_feat = atomic_feat

    @classmethod
    def from_excel(cls, filename):
        df = pd.read_excel(filename, index_col=0)
        return cls(df)

    def get_features(self, inputs, features):
        """
        Args:
            inputs (dict): keys are element symbols or 'T'; values
                are mol%
            features (list of str)

        Returns:
            pd.DataFrame
        """
        df = self.get_atomic_features(inputs, features)
        for key, val in inputs.items():
            df[key] = val

        return df

    def get_atomic_features(self, composition, features):
        available_features = set(self.atomic_feat.columns) & set(features)
        df = sum([
            self.atomic_feat.loc[element, available_features].values * fraction
            for element, fraction in composition.items()
            if element in self.atomic_feat.index
        ])
        return pd.DataFrame(df, columns=available_features)


def gen_non_linear(df, degree, features):
    """Generate non-linear features with the existing ones and
        the polynomial degree"""

    nonlinear = []
    for f1, f2 in itertools.combinations_with_replacement(features, degree):
        newfeature = f1 + '_' + f2
        df[newfeature] = df[f1] * df[f2]

        nonlinear.append(newfeature)

    return df, nonlinear
        

