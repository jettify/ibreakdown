import sys

import pandas as pd
import numpy as np
from tabulate import tabulate


class RegressionExplanation:
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
        contrib = [self.intercept] + self.contributions + [self.pred_value]

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


class ClassificationExplanation(RegressionExplanation):
    def _build_df(self):
        feature_names = [
            idx_to_name(idx, self._columns) for idx in self.feature_indexes
        ]
        feature_names = ['intercept'] + feature_names + ['PREDICTION']
        feature_values = [None] + self.feature_values + [None]

        data = {'Feature Name': feature_names, 'Feature Value': feature_values}
        contrib_columns = [
            f'Contribution:{c}' for c in enumerate(self._classes)
        ]
        contrib_val = np.concatenate(
            ([self.intercept], self.contributions, [self.pred_value])
        )
        data.update(zip(contrib_columns, contrib_val.T))
        df = pd.DataFrame(data)
        return df


def idx_to_name(idx, columns):
    if isinstance(idx, int):
        n = columns[idx]
    else:
        n = tuple(columns[i] for i in idx)
    return n
