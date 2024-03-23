import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import recall_score, make_scorer
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier

def xgboost_classifier_model(X: np.array, y: np.array):

    # Define XGBoostClassifier model
    xgb = XGBClassifier()

    # Wrap it with MultiOutputClassifier
    multi_xgb = MultiOutputClassifier(xgb)

    # Define the parameter grid for grid search
    param_grid = {
        'estimator__n_estimators': [50, 100, 200],
        'estimator__max_depth': [3, 6, 9],
        'estimator__learning_rate': [0.01, 0.1, 0.3],
        'estimator__subsample': [0.8, 1.0],
        'estimator__colsample_bytree': [0.8, 1.0],
        'estimator__gamma': [0, 1, 5]
    }

    # Create 3-fold cross-validation object
    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # Define custom scoring functions for recall
    custom_scoring = {'recall': make_scorer(recall_score, average='macro')}

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=multi_xgb,
                               param_grid=param_grid,
                               scoring=custom_scoring,
                               refit='recall',
                               cv=cv,
                               return_train_score=True)
    grid_search.fit(X, y)

    xgb_model = grid_search.best_estimator_

    return xgb_model
