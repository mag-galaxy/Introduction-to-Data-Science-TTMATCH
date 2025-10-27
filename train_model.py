import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ================================= parameters =================================
DATA_FILE = "train.csv"
TEST_FILE = "test.csv"
RESULT_FILE = "111504504_subission.csv"

# ================================= functions =================================
def predict_serverGetPoint(training_data, testing_data):
    # separate features and targets
    X = training_data.drop(columns=["serverGetPoint"])
    y = training_data["serverGetPoint"]
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # train
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    # model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    # model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # validate
    y_val_pred = model.fit(X_val, y_val)

    # predict testing set
    X_test = testing_data[X.columns]
    y_pred = model.predict(X_test)

    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return y_pred

def predict_pointId(data):
    return

def predict_actionId(data):
    return
    

# ================================= main =================================
# read data
df_data = pd.read_csv(DATA_FILE)
df_test = pd.read_csv(TEST_FILE)
df_result = pd.read_csv(RESULT_FILE)

df_result["serverGetPoint"] = predict_serverGetPoint(df_data, df_test)