import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import recall_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

def random_forest_classifier_model(X: np.array, y: np.array):

    # Define RandomForestClassifier model
    rf = RandomForestClassifier()

    # Wrap it with MultiOutputClassifier
    multi_rf = MultiOutputClassifier(rf)

    # Define the parameter grid for grid search
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [None, 10, 20],
        'estimator__min_samples_split': [2, 5, 10],
        'estimator__min_samples_leaf': [1, 2, 4],
        'estimator__bootstrap': [True, False]
    }

    # Create 3-fold cross-validation object
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # Define custom scoring functions for recall
    custom_scoring = {'recall': make_scorer(recall_score, average='macro')}

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=multi_rf,
                               param_grid=param_grid,
                               scoring=custom_scoring,
                               refit='recall',
                               cv=cv,
                               return_train_score=True)
    grid_search.fit(X, y)

    rf_model = grid_search.best_estimator_

    return rf_model
