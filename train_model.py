import pandas as pd
import numpy as np

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
TARGET = ["serverGetPoint", "actionId", "pointId"]
FEAT_NOT_USED = ["sex", "match", "numberGame", "rally_id"]
MY_MODELS = {}

# ================================= functions =================================
def TT_predict(training_data, testing_data, target):
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

def make_sequence_samples(df, target_cols, max_history=3):
    """
    把每個 rally_uid 展開成多筆樣本：
    用前 t-1 拍的特徵 → 預測第 t 拍的 target。
    """
    all_samples = []

    for rally_id, group in df.groupby("rally_uid"):
        group = group.sort_values("strickNum")
        # 從第2拍開始，每一拍都可當成「預測目標」
        for t in range(1, len(group)):
            current_row = group.iloc[t].copy()
            # 取前最多 max_history 拍
            history = group.iloc[max(0, t - max_history):t]
            # 把歷史拍的平均 or 最後值作為特徵
            features = history.mean(numeric_only=True).to_dict()
            # 或者取最後一拍的特徵（更常見）
            last_row = history.iloc[-1]
            for col in history.columns:
                features[f"prev_{col}"] = last_row[col]
            # 標籤
            for target in target_cols:
                features[target] = current_row[target]
            all_samples.append(features)

    df_out = pd.DataFrame(all_samples)
    return df_out

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

# save result (prediction)
df_result.to_csv(RESULT_FILE, index=False)