import pandas as pd
import sklearn
import argparse
import joblib
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ================================= arguments =================================
parser = argparse.ArgumentParser(description="TTMATCH model training")
parser.add_argument('-f', '--datafile', default="train.csv", help="training data file name")
parser.add_argument('-f', '--testfile', default="test.csv", help="testing data file name")
args = parser.parse_args()
DATA_FILE = args.datafile
TEST_FILE = args.testfile

# ================================= parameters =================================
RESULT_FILE = "111504504_subission.csv"

# ================================= functions =================================
def predict_serverGetPoint(training_data, testing_data, submission):
    # select features
    training_data = training_data.drop(columns=["label"])

    # separate features and targets
    X = training_data.drop(columns=["serverGetPoint"])
    y = training_data["serverGetPoint"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42, stratify=y)
    X_train = X
    y_train = y

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    X_test = testing_data
    y_pred = model.predict(X_test)

    submission["serverGetPoint"] = y_pred

    return

def predict_pointId(data):
    return

def predict_actionId(data):
    return
    

# ================================= main =================================
# read data
df_data = pd.read_csv(DATA_FILE)
df_test = pd.read_csv(TEST_FILE)
df_result = pd.read_csv(RESULT_FILE)

