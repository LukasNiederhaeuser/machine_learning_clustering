import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from typing import Tuple
from src.read_data import read_file

# Raw folder
FOLDER_PROCESSED = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'processed')


def read_input_data(filename: str, folder: str = 'raw') -> pd.DataFrame:

    df = read_file(folder=folder, filename=filename, delimiter=',')

    return df


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
    X = col_trans_predictor.fit_transform(data_predictors)
    # Get the column names after transformation for predictor variables
    X_columnnames = col_trans_predictor.named_transformers_['num_pipeline_predictor'].named_steps['scale'].get_feature_names_out(numerical_columns_predictor)

    # Fit and transform the data for the target variable
    y = col_trans_target.fit_transform(data_target.to_frame())
    # Get the column names after transformation for the target variable
    y_columnnames = col_trans_target.named_transformers_['target_pipeline'].named_steps['onehot'].get_feature_names_out(['species'])

    return X, X_columnnames, y, y_columnnames


def processed_data_to_dataframe(X: np.ndarray, X_columnnames: list, y:np.ndarray, y_columnnames:list) -> pd.DataFrame:

    # Create dataframes for features and target
    df_X = pd.DataFrame(data=X, columns=X_columnnames)
    df_y = pd.DataFrame(data=y, columns=y_columnnames)

    # Concatenate the DataFrames horizontally (axis=1)
    df_combined = pd.concat([df_X, df_y], axis=1)

    return df_combined


def store_data(df: pd.DataFrame, filename:str):
    
    
    try:
        # define file path and filename
        path = os.path.join(FOLDER_PROCESSED, filename)
        # save data to CSV
        df.to_csv(path, index=False)
        print(f"Successfully executed - processed {filename} data saved to: {path}")

    except Exception as e:
        print(f"An error occurred while saving the data: {e}")


def main():

    # process training data
    df_train = read_input_data(filename='iris_train.csv', folder='raw')
    X_train, X_train_columnnames, y_train, y_train_columnnames = data_preprocessing(data=df_train)
    df_train_combined = processed_data_to_dataframe(X_train, X_train_columnnames, y_train, y_train_columnnames)
    store_data(df=df_train_combined, filename='iris_train_processed.csv')

    # process test data
    df_test = read_input_data(filename='iris_test.csv', folder='raw')
    X_test, X_test_columnnames, y_test, y_test_columnnames = data_preprocessing(data=df_test)
    df_test_combined = processed_data_to_dataframe(X_test, X_test_columnnames, y_test, y_test_columnnames)
    store_data(df=df_test_combined, filename='iris_test_processed.csv')


if __name__ == '__main__':
    main()
