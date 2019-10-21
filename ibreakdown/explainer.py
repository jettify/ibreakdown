import numpy as np
from .utils import normalize_array, multiply_row, to_matrix


class RegressionExplainer:
    def __init__(self, clf):
        self._clf = clf
        self._data = None
        self._columns = None
        self._hints = None
        self._baseline = None

    def fit(self, data, columns, hints=None):
        self._data = data
        self._columns = columns
        self._hints = hints
        self._baseline = self._mean_predict(data).reshape(1, -1)

    def explain(self, row):
        instance = to_matrix(row)
        instance = normalize_array(instance)
        exp = self._explain_bottom_up(instance)
        return exp

    def _explain_bottom_up(self, instance):
        num_rows, num_features = self._data.shape
        important_variables = {}
        groups = features_groups(num_features)
        for group in groups:
            new_data = np.copy(self._data)
            new_data[:, group] = instance[:, group]
            pred_mean = self._mean_predict(new_data)
            if isinstance(group, int):
                important_variables[group] = pred_mean - self._baseline[0][0]
            else:
                important_variables[group] = (
                    pred_mean
                    - self._baseline[0][0]
                    - sum(important_variables[g] for g in group)
                )

        import pprint

        pprint.pprint(important_variables)
        preds = sorted(important_variables.items(), key=lambda v: abs(v[1]))
        import pprint

        pprint.pprint(preds)
        return None

    def _mean_predict(self, data):
        return self._clf.predict(data).mean(axis=0)


def features_groups(num_features):
    result = list(range(0, num_features))
    for i in range(0, num_features):
        for j in range(i + 1, num_features):
            result.append((i, j))
    return result


class ClassificationExplainer:
    pass
