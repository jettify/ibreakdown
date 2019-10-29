import io

import pytest
import numpy as np

from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from ibreakdown.explainer import RegressionExplainer, ClassificationExplainer


def assert_exp_invariant(exp, pred):
    invariant = np.sum(exp.contributions, axis=0) + exp.intercept
    assert invariant == pytest.approx(pred)


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
    for i in range(2):
        observation = X_test[i: i + 1]
        pred = clf.predict(observation)
        exp = explainer.explain(observation)
        assert_exp_invariant(exp, pred[0])

        exp = explainer.explain(observation, check_interactions=False)
        assert_exp_invariant(exp, pred[0])

        with io.StringIO() as buf:
            exp.print(file=buf, flush=True)
            assert len(buf.getvalue()) > 0


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

    for i in range(2):
        observation = X_test[i: i + 1]
        exp = explainer.explain(observation)
        pred = clf.predict_proba(observation)
        assert_exp_invariant(exp, pred[0])

        exp = explainer.explain(observation, check_interactions=False)
        assert_exp_invariant(exp, pred[0])

        with io.StringIO() as buf:
            exp.print(file=buf, flush=True)
            assert len(buf.getvalue()) > 0
