import pandas as pd
import numpy as np

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