import numpy as np

from .explanation import ClassificationExplanation, RegressionExplanation
from .utils import (
    normalize_array,
    to_matrix,
    magnituge,
    features_groups,
    feature_group_values,
)


class RegressionExplainer:

    exp_class = RegressionExplanation

    def __init__(self, clf):
        self._clf = clf
        self._data = None
        self._columns = None
        self._baseline = None

    def fit(self, data, columns=None, classes=None):
        self._data = data
        if columns is None:
            columns = list(range(data.shape[1]))

        self._columns = columns
        self._classes = classes or [0]
        self._baseline = self._mean_predict(data)

    def explain(self, row):
        instance = to_matrix(row)
        instance = normalize_array(instance)
        path = self._compute_explanation_path(instance)
        pred_value = self._mean_predict(instance)
        feature_indexes, featrue_values, contrib = self._explain_path(
            path, instance
        )

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
        return self._clf.predict(data).mean(axis=0)

    def _compute_explanation_path(self, instance):
        num_rows, num_features = self._data.shape
        important_variables = {}
        groups = features_groups(num_features)
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
                    - np.sum(important_variables[g] for g in group)
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
        featrue_values = feature_group_values(feature_indexes, instance)
        return feature_indexes, featrue_values, contrib


class ClassificationExplainer(RegressionExplainer):

    exp_class = ClassificationExplanation

    def _sort(self, important_variables):
        return sorted(important_variables.items(), key=lambda v: -magnituge(v))

    def _mean_predict(self, data):
        return self._clf.predict_proba(data).mean(axis=0)
