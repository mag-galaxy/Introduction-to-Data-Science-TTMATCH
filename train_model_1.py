# ================================= libraries =================================
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
# ================================= parameters =================================
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
RESULT_FILE = "111504504_submission.csv"
TARGET = ["serverGetPoint", "actionId", "pointId"]
FEAT_NOT_USED = ["rally_uid", "sex", "match", "numberGame", "rally_id", "strickNumber", "let"]
FEATURES = [
    "scoreSelf", "scoreOther", "gamePlayerId", "gamePlayerOtherId",
    "serveId", "serveNumber", "strickId", "handId", 
    "strengthId", "spinId", "pointId", "actionId", "positionId"
]
MAX_SEQ_LEN = 10
EPOCHS =15

# ================================= functions =================================
def make_sequences(df, is_train):
    X, y_server, y_action, y_point = [], [], [], []
    for rally_uid, group in df.groupby("rally_uid"):
        # print("processing rally_uid: ", rally_uid)
        group = group.sort_values("strickNumber")
        seq = group[FEATURES].values.tolist()
        length = len(seq)
        
        # 每個rally構造多筆樣本: 使用前k拍預測第k+1拍
        for k in range(1, length):
            hist_seq = seq[:k]
            hist_seq = pad_sequences([hist_seq[-MAX_SEQ_LEN:]], maxlen=MAX_SEQ_LEN, 
                                     padding='pre', truncating='pre', value=0)[0]
            X.append(hist_seq)
            
            if is_train:
                y_server.append(group.iloc[k]["serverGetPoint"])
                y_action.append(group.iloc[k]["actionId"])
                y_point.append(group.iloc[k]["pointId"])
    
    if is_train:
        return np.array(X), np.array(y_server), np.array(y_action), np.array(y_point)
    else:
        return np.array(X)

def build_lstm_model(num_features, num_classes, name):
    model = Sequential(name=name)
    model.add(Masking(mask_value=0, input_shape=(MAX_SEQ_LEN, num_features)))
    model.add(LSTM(128))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(1e-3), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ================================= main =================================
# read data
print("Loading data...")
df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)
df_submission = pd.read_csv(RESULT_FILE)

df_train["actionId"] = df_train["actionId"].replace(-1, 19)
df_test["actionId"] = df_test["actionId"].replace(-1, 19)
df_train["pointId"] = df_train["pointId"].replace(-1, 10)
df_test["pointId"] = df_test["pointId"].replace(-1, 10)

# group data with rally_uid
X_train, y_server, y_action, y_point = make_sequences(df_train, True)

num_features = len(FEATURES)
num_server_classes = len(np.unique(y_server))
num_action_classes = len(np.unique(y_action))
num_point_classes  = len(np.unique(y_point))

print("category count of server: ",num_server_classes)
print("category count of action: ", num_action_classes)
print("category count of point: ", num_point_classes)

# train
X_tr, X_val, y_server_tr, y_server_val = train_test_split(X_train, y_server, test_size=0.1, random_state=42)
_, _, y_action_tr, y_action_val = train_test_split(X_train, y_action, test_size=0.1, random_state=42)
_, _, y_point_tr, y_point_val = train_test_split(X_train, y_point, test_size=0.1, random_state=42)

print("Training serverGetPoint model...")
model_server = build_lstm_model(num_features, num_server_classes, "serverGetPoint")
model_server.fit(X_tr, y_server_tr, epochs=EPOCHS, batch_size=128, validation_data=(X_val, y_server_val))

print("Training actionId model...")
model_action = build_lstm_model(num_features, num_action_classes, "actionId")
model_action.fit(X_tr, y_action_tr, epochs=EPOCHS, batch_size=128, validation_data=(X_val, y_action_val))

print("Training pointId model...")
model_point = build_lstm_model(num_features, num_point_classes, "pointId")
model_point.fit(X_tr, y_point_tr, epochs=EPOCHS, batch_size=128, validation_data=(X_val, y_point_val))

rally_ids = []
pred_server, pred_action, pred_point = [], [], []

for rally_uid, group in df_test.groupby("rally_uid"):
    # 取出該 rally 的全部球，轉成序列
    seq = group[FEATURES].values

    # Padding 成固定長度
    seq_padded = pad_sequences([seq], maxlen=MAX_SEQ_LEN, dtype='float32', padding='pre', truncating='pre')

    # 各模型預測下一球
    ps = model_server.predict(seq_padded)
    pa = model_action.predict(seq_padded)
    pp = model_point.predict(seq_padded)

    # 取最大機率的類別
    pred_server.append(np.argmax(ps))
    pred_action.append(np.argmax(pa))
    pred_point.append(np.argmax(pp))

    rally_ids.append(rally_uid)

# check data length align
rally_ids = df_test["rally_uid"].unique()
print("len pred server: ", len(pred_server))
print("len pred action: ", len(pred_action))
print("len pred point: ", len(pred_point))
assert len(pred_server) == len(pred_action) == len(pred_point) == len(rally_ids), \
    f"❌ 預測結果長度不符：rally_ids={len(rally_ids)}, server={len(pred_server)}, action={len(pred_action)}, point={len(pred_point)}"

# write result into submission.csv
df_submission["serverGetPoint"] = pred_server
df_submission["actionId"] = pred_action
df_submission["pointId"] = pred_point

df_submission["actionId"] = df_submission["actionId"].replace(19, -1)
df_submission["pointId"] = df_submission["pointId"].replace(10, -1)

df_submission.to_csv(RESULT_FILE, index=False)
print("result saved to ", RESULT_FILE)
