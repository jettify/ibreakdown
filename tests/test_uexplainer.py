import numpy as np
from sklearn.datasets import load_boston
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ibreakdown import (
    URegressionExplainer,
    UClassificationExplainer,
)
from sklearn.ensemble import GradientBoostingRegressor


def assert_exp_invariant(exp, pred):
    invariant = np.sum(exp.contributions[0], axis=0) + exp.baseline
    assert np.allclose(invariant, pred)


def test_uregression(seed):
    boston = load_boston()
    columns = list(boston.feature_names)
    X, y = boston['data'], boston['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    clf = GradientBoostingRegressor(random_state=seed)
    clf.fit(X_train, y_train)
    explainer = URegressionExplainer(clf.predict)
    explainer.fit(X_train, columns)
    for i in range(5):
        observation = X_test[i: i + 1]
        exp = explainer.explain(observation)
        pred = clf.predict(observation)
        exp.print()
        exp.plot()

        assert_exp_invariant(exp, pred)


def test_uclassification(seed):
    iris = load_iris()
    columns = iris.feature_names
    X = iris.data
    y = iris.target
    classes = iris.target_names.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=seed
    )
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    explainer = UClassificationExplainer(clf.predict_proba)
    explainer.fit(X_train, columns, classes=classes)

    for i in range(2):
        observation = X_test[i: i + 1]
        exp = explainer.explain(observation)
        pred = clf.predict_proba(observation)

        exp = explainer.explain(observation)
        exp.print(classes[0])
        assert_exp_invariant(exp, pred)
