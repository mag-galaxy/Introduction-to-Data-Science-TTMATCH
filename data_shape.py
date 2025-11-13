import pandas as pd
import numpy as np

TRAIN_FILE = "train.csv"
FEATURES = [
    "serveId", "serveNumber", "strickId", "handId", 
    "strengthId", "spinId", "pointId", "actionId", "positionId"
]

def count(df, column):
    cnt = 0
    for i in range(len(df[column])):
        if df[column][i] == 1:
            cnt += 1
    return cnt

# FILE_NAME = "test.csv"
# test = pd.read_csv(FILE_NAME)
# print("Max strokes in a rally:", test["strickNumber"].max())
# print("count of strikNumber == 1: ", count(test, "strickNumber"))


X_train = np.load("X.npy")
y_server = np.load("y_server.npy")
y_action = np.load("y_action.npy")
y_point = np.load("y_point.npy")

print(X_train.shape)
print(y_server.shape)
print(y_action.shape)
print(y_point.shape)
print(y_action[:5])

df_tmp = pd.read_csv(TRAIN_FILE)
print(FEATURES)
vocab_sizes = [df_tmp[f].max() for f in FEATURES]  # 各特徵最大值+1
print(vocab_sizes)
vocab_sizes = [len(df_tmp[f].unique()) for f in FEATURES]
print(vocab_sizes, "\n\n")

df_tmp = pd.read_csv("test.csv")
print(FEATURES)
vocab_sizes = [df_tmp[f].max() for f in FEATURES]  # 各特徵最大值+1
print(vocab_sizes)
vocab_sizes = [len(df_tmp[f].unique()) for f in FEATURES]
print(vocab_sizes)