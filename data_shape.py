import pandas as pd

def count(df, column):
    cnt = 0
    for i in range(len(df[column])):
        if df[column][i] == 1:
            cnt += 1
    return cnt

FILE_NAME = "test.csv"
test = pd.read_csv(FILE_NAME)
print("Max strokes in a rally:", test["strickNumber"].max())
print("count of strikNumber == 1: ", count(test, "strickNumber"))

mod = pd.read_csv("111504504_submission.csv")
len = len(mod["rally_uid"])
print(len)

for i in range(len):
    if mod.loc[i, "actionId"] == -1 or mod.loc[i, "pointId"] == -1:
        mod.loc[i, "actionId"] = -1
        mod.loc[i, "pointId"] = -1

mod.to_csv("111504504_submission.csv", index=False)