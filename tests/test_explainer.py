import pytest
import numpy as np
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from ibreakdown.explainer import RegressionExplainer, features_groups


np.set_printoptions(suppress=True)


expected_columns_up = [
    'LSTAT',
    'PTRATIO',
    'CRIM',
    'B',
    'RM',
    'INDUS',
    'CHAS',
    'NOX',
    'TAX',
    'AGE',
    'ZN',
    'DIS',
    'RAD',
]

expected_contributions_up = np.array([])


@pytest.mark.parametrize(
    'exp_columns, exp_contributions',
    [(expected_columns_up, expected_contributions_up)],
)
def test_regression(seed, exp_columns, exp_contributions):
    boston = load_boston()
    columns = list(boston.feature_names)
    X, y = boston['data'], boston['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    clf = RandomForestRegressor(
        n_estimators=600, max_depth=2, random_state=seed, oob_score=True
    )
    clf.fit(X_train, y_train)

    observation = X_test[0]
    explainer = RegressionExplainer(clf)
    explainer.fit(X_train, columns)
    explainer.explain(observation)


def test_features_pairs():
    result = features_groups(3)
    assert result == [0, 1, 2, (0, 1), (0, 2), (1, 2)]
