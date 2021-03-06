import sys

import pandas as pd
import numpy as np
from tabulate import tabulate

from .utils import (
    feature_group_values,
    features_groups,
    magnituge,
    normalize_array,
    to_matrix,
)


class IRegressionExplanation:
    """Class provides access to prediction explanation data.
    """

    def __init__(
        self,
        pred_value,
        feature_indexes,
        feature_values,
        contributions,
        intercept,
        columns=None,
        classes=None,
    ):
        self.pred_value = pred_value
        self.feature_indexes = feature_indexes
        self.feature_values = feature_values
        self.contributions = contributions
        self.columns = columns
        self.intercept = intercept
        self.baseline = intercept
        self.digits = 4
        if columns is not None:
            self._columns = columns
        else:
            self._columns = [str(v) for v in range(len(feature_values))]
        if classes is None:
            self._classes = [str(i) for i in range(len(intercept))]
        else:
            self._classes = classes

    def _build_df(self):
        feature_names = [
            idx_to_name(idx, self._columns) for idx in self.feature_indexes
        ]
        feature_names = ['intercept'] + feature_names + ['PREDICTION']
        feature_values = [None] + self.feature_values + [None]
        contrib = (
            [self.intercept] + self.contributions.tolist() + [self.pred_value]
        )

        data = {
            'Feature Name': feature_names,
            'Feature Value': feature_values,
            'Contributions': contrib,
        }
        df = pd.DataFrame(data)
        return df

    def print(self, file=sys.stdout, flush=False):
        df = self._build_df()
        table = tabulate(df, tablefmt='psql', headers='keys')
        print(table, file=file, flush=flush)


class IClassificationExplanation(IRegressionExplanation):
    def _build_df(self):
        feature_names = [
            idx_to_name(idx, self._columns) for idx in self.feature_indexes
        ]
        feature_names = ['intercept'] + feature_names + ['PREDICTION']
        feature_values = [None] + self.feature_values + [None]

        data = {'Feature Name': feature_names, 'Feature Value': feature_values}
        contrib_columns = [f'Contrib:{c}' for c in self._classes]
        contrib_val = np.concatenate(
            ([self.intercept], self.contributions, [self.pred_value])
        )
        data.update(zip(contrib_columns, contrib_val.T))
        df = pd.DataFrame(data)
        return df


class IRegressionExplainer:

    exp_class = IRegressionExplanation

    def __init__(self, predict_func):
        self._predict_func = predict_func
        self._data = None
        self._columns = None
        self._baseline = None
        self._classes = None

    def fit(self, data, columns=None):
        self._data = data
        if columns is None:
            columns = list(range(data.shape[1]))

        self._classes = [0]
        self._columns = columns
        self._baseline = self._mean_predict(data)

    def explain(self, row, check_interactions=True):
        instance = to_matrix(row)
        instance = normalize_array(instance)
        path = self._compute_explanation_path(instance, check_interactions)
        pred_value = self._mean_predict(instance)
        feature_indexes, contrib = self._explain_path(path, instance)

        featrue_values = feature_group_values(feature_indexes, instance)
        return self.exp_class(
            pred_value,
            feature_indexes,
            featrue_values,
            contrib,
            self._baseline,
            self._columns,
            self._classes,
        )

    def _mean_predict(self, data):
        return self._predict_func(data).mean(axis=0)

    def _compute_explanation_path(self, instance, check_interactions=True):
        num_rows, num_features = self._data.shape
        important_variables = {}
        groups = features_groups(num_features, check_interactions)
        for group in groups:
            new_data = np.copy(self._data)
            new_data[:, group] = instance[:, group]
            pred_mean = self._mean_predict(new_data)
            if isinstance(group, int):
                impact = pred_mean - self._baseline
            else:
                impact = (
                    pred_mean
                    - self._baseline
                    - np.sum([important_variables[g] for g in group])
                )
            important_variables[group] = impact

        path = self._sort(important_variables)
        return path

    def _sort(self, important_variables):
        return sorted(important_variables.items(), key=lambda v: -abs(v[1]))

    def _explain_path(self, path, instance):
        _, num_features = self._data.shape
        features = set(range(num_features))
        new_data = np.copy(self._data)
        important_variables = []
        while features:
            group, _ = path.pop(0)
            if not set([group] if isinstance(group, int) else group).issubset(
                features
            ):
                continue
            new_data[:, group] = instance[:, group]
            pred_mean = self._mean_predict(new_data)
            important_variables.append((group, pred_mean))
            features.difference_update(
                set([group] if isinstance(group, int) else group)
            )

        cummulative = [v[1] for v in important_variables]
        feature_indexes = [v[0] for v in important_variables]
        contrib = np.diff(np.array([self._baseline] + cummulative), axis=0)
        return feature_indexes, contrib


class IClassificationExplainer(IRegressionExplainer):

    exp_class = IClassificationExplanation

    def _sort(self, important_variables):
        return sorted(important_variables.items(), key=lambda v: -magnituge(v))

    def fit(self, data, columns=None, classes=None):
        self._classes = classes or [0]
        super().fit(data, columns=columns)


def idx_to_name(idx, columns):
    if isinstance(idx, int):
        n = columns[idx]
    else:
        n = tuple(columns[i] for i in idx)
    return n
