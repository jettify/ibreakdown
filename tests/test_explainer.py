import pytest
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from ibreakdown.explainer import (
    RegressionExplainer,
    ClassificationExplainer,
    features_groups,
)


def test_regression(seed):
    boston = load_boston()
    columns = list(boston.feature_names)
    X, y = boston['data'], boston['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    observation = X_test[0]
    clf = RandomForestRegressor(
        n_estimators=600, max_depth=2, random_state=seed, oob_score=True
    )
    clf.fit(X_train, y_train)
    explainer = RegressionExplainer(clf)
    explainer.fit(X_train, columns)
    for i in range(3):
        observation = X_test[i: i + 1]
        pred = clf.predict(observation)
        exp = explainer.explain(observation)
        # check invariant
        assert sum(exp.contributions) + exp.intercept == pytest.approx(pred[0])
        exp.print()


def test_features_pairs():
    result = features_groups(3)
    assert result == [0, 1, 2, (0, 1), (0, 2), (1, 2)]


@pytest.mark.skip(reason='WIP')
def test_multiclass(seed):
    iris = load_iris()
    columns = iris.feature_names
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed
    )

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    explainer = ClassificationExplainer(clf)
    explainer.fit(X_train, columns)
    exp = explainer.explain(X_test[0])
    exp.print()
