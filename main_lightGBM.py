import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt
import os

VIS_PATH = "visualization/"
if not os.path.exists(VIS_PATH):
    os.makedirs(VIS_PATH)

# read train.csv and test.csv
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


# print(train_df.head())
# print(train_df.info())

# categorical features: survived, Sex, Pclass, Embarked, Cabin, Name, Ticket, Sibsp, Parch
# numerical features: Age, Fare, passengerId

PassengerID = test_df['PassengerId']

# train test まとめて前処理したいのでまとめる
train_df.drop(['Survived'], axis=1, inplace=True)
combined_df = train_df.append(test_df)
# ch

categorical_val_1 = ["Survived", "Pclass", "Sex", "SibSp", "Parch", "Embarked"]
# visualize the frequency of each categorical features
for col in categorical_val_1:
    plt.figure(figsize=(10, 5))
    plt.title(col)
    train_df[col].value_counts().plot.bar()
    plt.savefig(VIS_PATH + col + ".png")

categorical_val_2 = ["Cabin", "Name", "Ticket"]
# print the count of each value
for col in categorical_val_2:
    print(train_df[col].value_counts())