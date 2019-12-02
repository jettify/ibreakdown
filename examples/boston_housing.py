from ibreakdown.explainer import IRegressionExplainer
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


def read_data(seed=None):
    boston = load_boston()
    columns = list(boston.feature_names)
    X, y = boston['data'], boston['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    return X_train, X_test, y_train, y_test, columns


def train_model(X_train, y_train, seed=None):
    rf = RandomForestRegressor(random_state=seed)
    param_grid = {
        'n_estimators': [300],
        'max_depth': [3],
        'oob_score': [True],
    }

    gs = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
    )
    gs.fit(X_train, y_train)
    return gs.best_estimator_


def main():
    seed = 42
    X_train, X_test, y_train, y_test, columns = read_data(seed=seed)
    model = train_model(X_train, y_train, seed=seed)

    explainer = IRegressionExplainer(model.predict)
    explainer.fit(X_train, columns)

    for i in range(2):
        observation = X_test[i: i + 1]
        model.predict(observation)
        exp = explainer.explain(observation)
        # Do not calculate interactions for faster results
        exp = explainer.explain(observation, check_interactions=False)
        exp.print()


if __name__ == '__main__':
    main()
