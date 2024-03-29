import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple

# Raw folder
FOLDER_RAW = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
FILENAME_RAW = 'iris_raw.csv'

# Filename train and test
FILENAME_TRAIN = 'iris_train.csv'
FILENAME_TEST = 'iris_test.csv'


def load_data() -> pd.DataFrame:
    
    # Load data
    data = pd.read_csv(os.path.join(FOLDER_RAW, FILENAME_RAW), delimiter=",")
    
    return data


def shuffle_and_stratified_split(data: pd.DataFrame, test_size: float = 0.2, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    
    # shuffle the data randomly
    shuffled_data = data.sample(frac=1, random_state=random_seed)

    # Stratified split based on the 'species' column
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    for train_index, test_index in split.split(shuffled_data, shuffled_data['species']):
        train_set = shuffled_data.iloc[train_index]
        test_set = shuffled_data.iloc[test_index]

    return train_set, test_set


def split_predictor_and_target(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list, list]:

    # create copy
    df_copy = df.copy()

    # define columns
    df_predictors = df_copy.iloc[:, :4]
    df_targets = df_copy.iloc[:, 4:]

    # get column names of dataframes
    columns_X = df_predictors.columns
    columns_y = df_targets.columns

    # extracting numpy arrays from dataframe
    X_array = df_predictors.values
    y_array = df_targets.values

    return X_array, y_array, columns_X, columns_y


def store_data(strat_train_set, strat_test_set):
        
    try:
        # define file paths for train and test sets
        train_set_path = os.path.join(FOLDER_RAW, FILENAME_TRAIN)
        test_set_path = os.path.join(FOLDER_RAW, FILENAME_TEST)

        # save train and test sets to CSV
        strat_train_set.to_csv(train_set_path, index=False)
        strat_test_set.to_csv(test_set_path, index=False)

        print(f"Successfully executed - stratified train and test set saved to: {train_set_path}")

    except Exception as e:
        print(f"An error occurred while saving the data: {e}")


def main():

    # load data
    data = load_data()
    # Create train and test split
    strat_train_set, strat_test_set = shuffle_and_stratified_split(data=data, test_size=0.2, random_seed=42)
    # Save train and test split
    store_data(strat_train_set, strat_test_set)

if __name__ == '__main__':
    main()
