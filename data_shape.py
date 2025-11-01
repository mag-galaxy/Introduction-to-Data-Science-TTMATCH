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
