import pandas as pd
import numpy

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ================================= parameters =================================
DATA_FILE = "train.csv"
TEST_FILE = "test.csv"
RESULT_FILE = "111504504_submission.csv"

# ================================= functions =================================
def TT_predict(training_data, testing_data):
    training_data = training_data.sort_values(["rally_uid", "strokeNum"]).reset_index(drop=True)

    # separate features and targets
    X = training_data.drop(columns=["serverGetPoint"])
    y = training_data["serverGetPoint"]
    # le = LabelEncoder()
    # y_encoded = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    # model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # validate
    y_val_pred = model.predict(X_val)

    # predict testing set
    X_test = testing_data[X.columns]
    y_pred = model.predict(X_test)

    # print(classification_report(y_val, y_val_pred))

    return y_pred

# ================================= main =================================
# read data
df_train = pd.read_csv(DATA_FILE)
df_test = pd.read_csv(TEST_FILE)
df_result = pd.read_csv(RESULT_FILE)

# check data shape
print("train.csv shape: ", df_train.shape)
print("train.csv headers: ", df_train.head())
print("test.csv shape: ", df_test.shape)
print("test.csv headers: ", df_test.head())
print("submission.csv shape: ", df_result.shape)
print("submission.csv headers: ", df_result.head())

# call functions for training and predict
print("start processing...")

df_result["serverGetPoint"] = TT_predict(df_train, df_test)
# df_result["pointId"] = predict_serverGetPoint(df_train, df_test)
# df_result["actionId"] = predict_serverGetPoint(df_train, df_test)

# save result (prediction)
df_result.to_csv(RESULT_FILE)