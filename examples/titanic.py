import pandas as pd

from ibreakdown import ClassificationExplainer
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def read_dataset(columns, seed=None):
    url = (
        'https://web.stanford.edu/class/archive/'
        'cs/cs109/cs109.1166/stuff/titanic.csv'
    )
    df = pd.read_csv(url)
    y = df['Survived']
    X = df[columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    return X_train, X_test, y_train, y_test


def build_model(num_features, cat_features, seed=None):
    preprocess = make_column_transformer(
        (make_pipeline(SimpleImputer(), StandardScaler()), num_features),
        (OneHotEncoder(categories='auto'), cat_features),
    )
    model = make_pipeline(
        preprocess, RandomForestClassifier(random_state=seed)
    )
    return model


def main():
    seed = 42
    columns = [
        'Age',  # num
        'Fare',  # num
        'Siblings/Spouses Aboard',  # num
        'Parents/Children Aboard',  # num
        'Pclass',  # cat
        'Sex',  # cat
    ]
    X_train, X_test, y_train, y_test = read_dataset(columns, seed)
    rf = build_model([0, 1, 2, 3], [4, 5], seed=seed)

    param_grid = {
        'columntransformer__pipeline__simpleimputer__strategy': ['mean'],
        'randomforestclassifier__min_samples_leaf': [5],
        'randomforestclassifier__min_samples_split': [12],
        'randomforestclassifier__n_estimators': [100],
    }

    gs = GridSearchCV(
        estimator=rf, param_grid=param_grid, scoring='roc_auc', cv=3, n_jobs=-1
    )
    gs.fit(X_train, y_train)
    print('-' * 100)
    print(gs.best_score_)
    print(gs.best_params_)
    print('-' * 100)
    class_map = ['Deceased', 'Survived']
    classes = [class_map[i] for i in gs.classes_]
    explainer = ClassificationExplainer(gs)
    explainer.fit(X_train, columns, classes)

    for i in range(10):
        observation = X_test[i: i + 1]
        exp = explainer.explain(observation)
        exp.print()


if __name__ == '__main__':
    main()
