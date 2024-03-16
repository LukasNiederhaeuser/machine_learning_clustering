import numpy as np

from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import recall_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multioutput import MultiOutputClassifier


def knn_classifier_model(X: np.array, y: np.array):

    # Define KNeighborsClassifier model
    knn = KNeighborsClassifier()

    # Wrap it with MultiOutputClassifier
    multi_knn = MultiOutputClassifier(knn)

    # Define the parameter grid for grid search
    param_grid = {
        'estimator__n_neighbors': [1, 2, 3, 4, 5],
        'estimator__weights': ['uniform', 'distance'],
        'estimator__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
    }

    # Create 3-fold cross-validation object
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # Define custom scoring functions for precision and recall
    custom_scoring = {'recall': make_scorer(recall_score, average='macro')}

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=multi_knn,
                               param_grid=param_grid,
                               scoring=custom_scoring,
                               refit='recall',
                               cv=cv,
                               return_train_score=True)
    grid_search.fit(X, y)

    knn_model = grid_search.best_estimator_

    return knn_model

