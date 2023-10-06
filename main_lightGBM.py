import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

def convert_obj_to_category(df_):
    object_cols = [col for col in df_.columns if df_[col].dtype == "object"]
    for col in object_cols:
        df_[col] = df_[col].astype("category")
    return df_

if __name__ == '__main__':

    VIS_PATH = "visualization/"
    if not os.path.exists(VIS_PATH):
        os.makedirs(VIS_PATH)

    # read train.csv and test.csv
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    PassengerID = test_df['PassengerId']

    X = train_df.drop(['Survived'], axis=1)
    X = convert_obj_to_category(X)
    y = train_df['Survived']
    X_test = convert_obj_to_category(test_df)

    # split train_df into train and valid with train_test_split
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=0)
    print(X_train.shape())