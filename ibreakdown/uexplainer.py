import numpy as np
import sys
import pandas as pd

from tabulate import tabulate
import matplotlib.pyplot as plt

from .utils import normalize_array, to_matrix


class URegressionExplanation:
    def __init__(
        self, prediction, observation, contributions, baseline, columns=None
    ):
        self.prediction = prediction
        self.observation = observation[0]
        self.contributions = contributions
        self.baseline = baseline
        if columns is not None:
            self._columns = columns
        else:
            self._columns = [str(v) for v in range(len(observation))]

    def _build_df(self):
        feature_names = self._columns
        feature_names = ['intercept'] + feature_names + ['PREDICTION']
        feature_values = [None] + self.observation.tolist() + [None]
        contrib = (
            [self.baseline]
            + self.contributions[0].tolist()
            + [self.prediction]
        )

        data = {
            'Feature Name': feature_names,
            'Feature Value': feature_values,
            'Contributions': contrib,
            'Contributions STD': [0]
            + np.std(self.contributions, axis=0).tolist()
            + [0],
        }
        df = pd.DataFrame(data)
        return df

    def print(self, file=sys.stdout, flush=False):
        df = self._build_df()
        table = tabulate(df, tablefmt='psql', headers='keys')
        print(table, file=file, flush=flush)

    def plot(self):
        fig1, ax1 = plt.subplots()
        ax1.set_title('Feature Contributions')
        y_axis = self.contributions[0]
        yerr = np.std(self.contributions, axis=0)
        x_axis = self._columns
        ax1.bar(x_axis, y_axis, yerr=yerr)
        fig1.savefig(f'foo_.png')


class URegressionExplainer:

    exp_class = URegressionExplanation

    def __init__(self, predict_func, sample_size=7, seed=None):
        self._predict_func = predict_func
        self._data = None
        self._columns = None
        self._baseline = None
        self._classes = None
        self._rand = np.random.RandomState(seed)
        self._sample_size = sample_size

    def fit(self, data, columns=None):
        self._data = data
        if columns is None:
            columns = list(range(data.shape[1]))

        self._classes = [0]
        self._columns = columns
        self._baseline = self._mean_predict(data)

    def explain(self, row, check_interactions=True):
        observation = to_matrix(row)
        observation = normalize_array(observation)
        pred_value = self._mean_predict(observation)
        main_path = self._compute_explanation_path(observation)
        pathes = [main_path]
        _, num_features = self._data.shape
        for _ in range(self._sample_size):
            path = main_path.copy()
            path = self._rand.permutation(num_features)
            pathes.append(path)

        contributions_stats = []
        for p in pathes:
            contrib = self._explain_path(p, observation)
            contributions_stats.append(contrib)

        contributions = np.array(contributions_stats)
        exp = self.exp_class(
            pred_value,
            observation,
            contributions,
            self._baseline,
            columns=self._columns,
        )
        return exp

    def _mean_predict(self, data):
        return self._predict_func(data).mean(axis=0)

    def _make_zeros(self):
        return np.zeros(self._data.shape[1])

    def _sort(self, feature_impact):
        p = np.argsort(np.abs(feature_impact), axis=0)[::-1]
        return p.reshape(1, -1)[0]

    def _compute_explanation_path(self, instance):
        num_rows, num_features = self._data.shape
        features = np.arange(num_features)

        feature_impact = self._make_zeros()

        for feature_idx in features:
            new_data = np.copy(self._data)
            new_data[:, feature_idx] = instance[:, feature_idx]
            pred_mean = self._mean_predict(new_data)
            feature_impact[feature_idx] = pred_mean
        return self._sort(feature_impact)

    def _explain_path(self, path, instance):
        _, num_features = self._data.shape
        new_data = np.copy(self._data)
        pred_mean = self._make_zeros()
        for i, feature_idx in enumerate(path):
            new_data[:, feature_idx] = instance[:, feature_idx]
            pred_mean[i] = self._mean_predict(new_data)
        means = np.insert(pred_mean, 0, self._baseline, axis=0)
        contributions = np.diff(np.array(means), axis=0)
        return contributions[np.argsort(path)]


class UClassificationExplainer(URegressionExplainer):

    def _make_zeros(self):
        return np.zeros((self._data.shape[1], self._baseline.shape[0]))

    def _sort(self, feature_impact):
        p = np.argsort(np.linalg.norm(feature_impact, axis=1))[::-1]
        return p.reshape(1, -1)[0]
