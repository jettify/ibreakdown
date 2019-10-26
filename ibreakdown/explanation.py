import sys

from terminaltables import AsciiTable


class Explanation:
    """Class provides access to prediction explanation data.
    """

    def __init__(
        self,
        feature_indexes,
        feature_values,
        contributions,
        intercept,
        columns=None,
    ):
        self.feature_indexes = feature_indexes
        self.feature_values = feature_values
        self.contributions = contributions.tolist()
        self.columns = columns
        self.intercept = intercept
        self.digits = 4
        if columns is not None:
            self._columns = columns
        else:
            self._columns = [str(v) for v in range(len(feature_values))]

    def _make_table_data(self,):
        table_data = []
        header = ('Feature', 'Value', 'Contribution')
        table_data.append(header)

        baseline = ('baseline', '-', str(self.intercept))
        table_data.append(baseline)
        for i, feature_idx in enumerate(self.feature_indexes):
            if isinstance(i, int):
                row = (
                    self._columns[feature_idx],
                    self.feature_values[i],
                    self.contributions[i],
                )
            else:
                row = (
                    (
                        self._columns[feature_idx[0]],
                        self._columns[feature_idx[0]],
                    ),
                    self.feature_values[i],
                    self.contributions[i],
                )
            table_data.append(row)
        return table_data

    def print(self, file=sys.stdout, flush=False):
        table_data = self._make_table_data()
        table = AsciiTable(table_data)
        print(table.table, file=file, flush=flush)

    def raw_result(self):
        return self.data
