import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve, validation_curve, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# Create table for missing data analysis
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data

df = pd.read_csv("train.csv")
df_raw = df.copy()

def clean_data(df):
    # fill missing data
    df.drop(['Cabin'], axis=1, inplace=True)
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    df.drop(df[df["Embarked"].isnull()].index, inplace=True)

    # create baseline
    df.drop(["PassengerId"], axis=1, inplace=True)
    df["Sex"] = pd.Categorical(df['Sex'])
    df["Embarked"] = pd.Categorical(df['Embarked'])
    df["Family_Size"] = df["SibSp"] + df["Parch"]
    df.drop(["SibSp", "Parch", "Name", "Ticket"], axis=1, inplace=True)

    df = pd.get_dummies(df, drop_first=True)
    return df

df = clean_data(df)
X = df.drop(["Survived"], axis=1, inplace=False)
Y = df["Survived"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

# fit logistic regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, Y_train)
predictions = logreg.predict(X_test)
# print("Accuracy = ", accuracy_score(Y_test, predictions))

submission_df = pd.read_csv("test.csv")
submission_df_raw = submission_df.copy()
submission_df = clean_data(submission_df)
submission_df["Fare"].fillna(submission_df["Fare"].mean(), inplace=True)
submission_pred = logreg.predict(submission_df)
submission_df = pd.DataFrame({
    "PassengerId": submission_df_raw["PassengerId"].values,
    "Survived": submission_pred
})
submission_df.to_csv("./submission.csv", index = False)