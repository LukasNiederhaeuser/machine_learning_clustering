import os
import joblib
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, KFold
from src.read_data import read_file
from typing import Tuple

# Define folders
FOLDER_PROCESSED = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')
FOLDER_RAW = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
FOLDER_MODEL = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')


def read_input_data(filename: str, folder: str = 'raw') -> pd.DataFrame:

    # Read input data        
    if folder == "raw":
        data = pd.read_csv(os.path.join(folder, filename), delimiter=',')
    elif folder == "processed":
        data = pd.read_csv(os.path.join(folder, filename), delimiter=',')
    else:
        print("Folder can be either: 'raw' or 'processed'")
        
    return data


def data_preprocessing(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    
    # Create a copy of the data
    data_copy = data.copy()

    # Create predictor and target columns
    data_predictors = data_copy.drop('species', axis=1)
    data_target = data_copy['species']

    # Get names of predictor variables
    numerical_columns_predictor = data_predictors.columns

    # Pipeline for numerical features
    num_pipeline_predictor = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scale', StandardScaler())
    ])

    # Pipeline for target variable (one-hot encoding)
    target_pipeline = Pipeline(steps=[
        ('onehot', OneHotEncoder())
    ])

    # Create column transformer for predictor variables
    col_trans_predictor = ColumnTransformer(transformers=[
        ('num_pipeline_predictor', num_pipeline_predictor, numerical_columns_predictor)
    ],
        remainder='passthrough',
        n_jobs=-1
    )

    # Create column transformer for target variable
    col_trans_target = ColumnTransformer(transformers=[
        ('target_pipeline', target_pipeline, ['species'])
    ],
        remainder='drop',
        n_jobs=-1
    )

    # Fit and transform the data for predictor variables
    transformed_data_predictor = col_trans_predictor.fit_transform(data_predictors)
    # Get the column names after transformation for predictor variables
    transformed_column_names_predictor = col_trans_predictor.named_transformers_['num_pipeline_predictor'].named_steps['scale'].get_feature_names_out(numerical_columns_predictor)

    # Fit and transform the data for the target variable
    transformed_data_target = col_trans_target.fit_transform(data_target.to_frame())
    # Get the column names after transformation for the target variable
    transformed_column_names_target = col_trans_target.named_transformers_['target_pipeline'].named_steps['onehot'].get_feature_names_out(['species'])

    return transformed_data_predictor, transformed_column_names_predictor, transformed_data_target, transformed_column_names_target


def train_random_forest_regressor(X, y):

    # Define Linear Regression model
    forest_reg = RandomForestRegressor()

    # Define the parameter grid for grid search
    param_grid = [
        {'n_estimators': [10, 20, 30, 40], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
        ]

    # Create 5-fold cross-validation object
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Perform GridSearchCV
    grid_search = GridSearchCV(estimator=forest_reg,
                               param_grid=param_grid,
                               scoring='neg_mean_squared_error',
                               cv=cv,
                               return_train_score=True)
    grid_search.fit(X, y)

    # Print the best parameters from grid search
    print("Best Parameters: ", grid_search.best_params_)

    # Get the best model from the grid search
    best_model = grid_search.best_estimator_

    # Define storage folder
    save_model_path = os.path.join(FOLDER_REG, "regression_model.joblib")

    # Store model 
    joblib.dump(best_model, save_model_path)
    print("Regression model saved successfully")