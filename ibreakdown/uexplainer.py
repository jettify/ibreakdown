import numpy as np
import sys
import pandas as pd

from tabulate import tabulate
from .utils import normalize_array, to_matrix

try:
    import seaborn as sns
    import matplotlib.pyplot as plt
except ImportError:
    plt, sns = None, None


_FEATURE_NAME = 'Feature Name'
_FEATURE_VALUE = 'Feature Value'
_CONTRIBUTION = 'Contribution'
_CONTRIBUTION_STD = 'Contribution STD'


class URegressionExplanation:
    def __init__(
        self,
        prediction,
        observation,
        contributions,
        baseline,
        columns=None,
        classes=None,
    ):
        self.prediction = prediction
        self.observation = observation
        self.contributions = contributions
        self.baseline = baseline
        if columns is not None:
            self._columns = columns
        else:
            self._columns = [str(v) for v in range(len(observation))]
        self._classes = None

    def _build_df(self):
        data = {
            _FEATURE_NAME: self._columns,
            _FEATURE_VALUE: self.observation,
            _CONTRIBUTION: self.contributions[0],
            _CONTRIBUTION_STD: np.std(self.contributions, axis=0),
        }
        df = pd.DataFrame(data)
        return df

    @property
    def df(self):
        return self._build_df()

    def print(self, file=sys.stdout, flush=False):
        df = self._build_df()
        table = tabulate(df, tablefmt='psql', headers='keys')
        print(table, file=file, flush=flush)

    def plot(self):
        if sns is None or plt is None:
            raise RuntimeError('Please install seaborn plotting library')

        sns.set(style='whitegrid')
        df = self._build_df()
        df = df.iloc[np.argsort(np.abs(df[_CONTRIBUTION].values))[::-1]]
        f, ax = plt.subplots(figsize=(13, 15))

        fn = df[_FEATURE_NAME].astype(str)
        fv = df[_FEATURE_VALUE].astype(str)
        y = '[' + fn + ']: ' + fv

        sns.set_color_codes('muted')
        b = sns.barplot(
            x=df[_CONTRIBUTION],
            y=y,
            data=df,
            label=_CONTRIBUTION,
            color='b',
            xerr=df[_CONTRIBUTION_STD],
        )

        title = 'Predicted: [{}]\nBaseline: [{}]'.format(
            self.prediction, self.baseline
        )
        b.axes.set_title(title, fontsize=20)
        ax.legend(ncol=2, loc='lower right', frameon=True)
        ax.set(ylabel=_FEATURE_NAME, xlabel=_FEATURE_VALUE)
        sns.despine(left=True, bottom=True)
        return f


class UClassificationExplanation(URegressionExplanation):
    def __init__(
        self,
        prediction,
        observation,
        contributions,
        baseline,
        columns=None,
        classes=None,
    ):
        super().__init__(
            prediction,
            observation,
            contributions,
            baseline,
            columns=columns,
            classes=classes,
        )
        self._classes = classes or list(range(len(self.baseline)))

    def _build_df(self, class_idx=None):
        data = {
            _FEATURE_NAME: self._columns,
            _FEATURE_VALUE: self.observation,
        }
        if class_idx is not None:
            classes = [(class_idx, self._classes[class_idx])]
        else:
            classes = [(i, c) for i, c in enumerate(self._classes)]

        for class_idx, class_name in classes:
            c = _CONTRIBUTION + f'\n {class_name}'
            c_std = _CONTRIBUTION_STD + f'\n {class_name}'
            data[c] = self.contributions[0][:, class_idx]
            data[c_std] = np.std(self.contributions, axis=0)[:, class_idx]

        df = pd.DataFrame(data)
        return df

    def print(self, class_=None, file=sys.stdout, flush=False):
        if class_ is None:
            class_idx = None
        else:
            class_idx = self._classes.index(class_)

        df = self._build_df(class_idx)
        table = tabulate(df, tablefmt='psql', headers='keys')
        print(table, file=file, flush=flush)

    def plot(self, class_=None):
        if sns is None or plt is None:
            raise RuntimeError('Please install seaborn plotting library')
        if class_ is None:
            class_idx = None
        else:
            class_idx = self._classes.index(class_)

        class_name = self._classes[class_idx]
        c = _CONTRIBUTION + f'\n {class_name}'
        c_std = _CONTRIBUTION_STD + f'\n {class_name}'
        sns.set(style='whitegrid')
        df = self._build_df(class_idx)
        df = df.iloc[np.argsort(np.abs(df[c].values))[::-1]]
        f, ax = plt.subplots(figsize=(10, 10))

        fn = df[_FEATURE_NAME].astype(str)
        fv = df[_FEATURE_VALUE].astype(str)
        y = '[' + fn + ']: ' + fv

        sns.set_color_codes('muted')
        b = sns.barplot(
            x=df[c],
            y=y,
            data=df,
            label=c,
            color='b',
            xerr=df[c_std],
        )

        title = 'Predicted: [{}]\nBaseline: [{}]'.format(
            self.prediction, self.baseline
        )
        b.axes.set_title(title, fontsize=20)
        ax.legend(ncol=2, loc='lower right', frameon=True)
        ax.set(ylabel=_FEATURE_NAME, xlabel=_FEATURE_VALUE)
        sns.despine(left=True, bottom=True)


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

        self._columns = columns
        self._baseline = self._mean_predict(data)

    def explain(self, row):
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
            observation[0],
            contributions,
            self._baseline,
            columns=self._columns,
            classes=self._classes,
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

    exp_class = UClassificationExplanation

    def __init__(self, predict_func, sample_size=7, seed=None):
        super().__init__(predict_func, sample_size=sample_size, seed=seed)
        self._classes = None

    def _make_zeros(self):
        return np.zeros((self._data.shape[1], self._baseline.shape[0]))

    def _sort(self, feature_impact):
        p = np.argsort(np.linalg.norm(feature_impact, axis=1))[::-1]
        return p.reshape(1, -1)[0]

    def fit(self, data, columns=None, classes=None):
        super().fit(data, columns=columns)
        self._classes = classes
