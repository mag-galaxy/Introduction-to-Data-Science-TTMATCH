import pandas as pd

FILE_NAME = "test.csv"
test = pd.read_csv(FILE_NAME)
print("Max strokes in a rally:", test["strickNumber"].max())
