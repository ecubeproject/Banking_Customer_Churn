import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None, lcols=None, ohecols=None, tcols=None, reduce_df=False):
        if isinstance(cols, str):
            self.cols = [cols]
        else:
            self.cols = cols

        if isinstance(lcols, str):
            self.lcols = [lcols]
        else:
            self.lcols = lcols

        if isinstance(ohecols, str):
            self.ohecols = [ohecols]
        else:
            self.ohecols = ohecols

        if isinstance(tcols, str):
            self.tcols = [tcols]
        else:
            self.tcols = tcols

        self.reduce_df = reduce_df

    def fit(self, X, y):
        if self.cols is None:
            self.cols = [c for c in X if str(X[c].dtype) == 'object']

        if self.lcols is None:
            self.lcols = [c for c in self.cols if X[c].nunique() <= 2]

        if self.ohecols is None:
            self.ohecols = [c for c in self.cols if 2 < X[c].nunique() <= 10]

        if self.tcols is None:
            self.tcols = [c for c in self.cols if X[c].nunique() > 10]

        self.lmaps = {col: dict(zip(X[col].values, X[col].astype('category').cat.codes.values)) for col in self.lcols}
        self.ohemaps = {col: X[col].unique()[:-1] if self.reduce_df else X[col].unique() for col in self.ohecols}
        self.global_target_mean = y.mean().round(2)
        self.sum_count = {
            col: {unique: (y[X[col] == unique].sum(), (X[col] == unique).sum()) for unique in X[col].unique()}
            for col in self.tcols
        }
        return self

    def transform(self, X, y=None):
        Xo = X.copy()
        for col, lmap in self.lmaps.items():
            Xo[col] = Xo[col].map(lmap).fillna(-1)

        for col, vals in self.ohemaps.items():
            for val in vals:
                Xo[f'{col}_{val}'] = (Xo[col] == val).astype('uint8')
            del Xo[col]

        if y is None:
            for col in self.sum_count:
                Xo[col] = X[col].map(
                    {cat: sum_count[0] / sum_count[1] for cat, sum_count in self.sum_count[col].items()}
                ).fillna(self.global_target_mean)
        else:
            for col in self.sum_count:
                vals = np.full(X.shape[0], np.nan)
                for cat, sum_count in self.sum_count[col].items():
                    ix = X[col] == cat
                    if sum_count[1] > 1:
                        vals[ix] = (sum_count[0] - y[ix]) / (sum_count[1] - 1)
                    else:
                        vals[ix] = (y.sum() - y[ix]) / (X.shape[0] - 1)
                # Assuming vals is a numpy array, convert it to a pandas Series
                Xo[col] = pd.Series(vals).fillna(self.global_target_mean)
        return Xo

    def fit_transform(self, X, y=None):
        """Fit and transform the data via label/one-hot/target encoding.

        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing columns to encode
        y : pandas Series, shape = [n_samples]
            Target values (required!).

        Returns
        -------
        pandas DataFrame
            Input DataFrame with transformed columns
        """

        return self.fit(X, y).transform(X, y)

class AddFeatures(BaseEstimator):
    def __init__(self, eps=1e-6):
        self.eps = eps

    def fit(self):
        return self

    def transform(self, X):
        Xo = X.copy()
        Xo['bal_per_product'] = Xo.Balance / (Xo.NumOfProducts + self.eps)
        Xo['bal_by_est_salary'] = Xo.Balance / (Xo.EstimatedSalary + self.eps)
        Xo['tenure_age_ratio'] = Xo.Tenure / (Xo.Age + self.eps)
        Xo['age_surname_enc'] = np.sqrt(Xo.Age) * Xo.Surname
        return Xo

    def fit_transform(self):
        """
        Parameters
        ----------
        X : pandas DataFrame, shape [n_samples, n_columns]
            DataFrame containing base columns using which new interaction-based features can be engineered
        """
        return self.fit(X, y).transform(X)