# ================================= libraries =================================
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

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
    "serveId", "serveNumber", "strickId", "handId", 
    "strengthId", "spinId", "pointId", "actionId", "positionId"
]
MAX_SEQ_LEN = 8
EPOCHS = 30

# ================================= functions =================================
def make_sequences(df):
    X, y_server, y_action, y_point = [], [], [], []
    for rally_uid, group in df.groupby("rally_uid"):
        # seq is features of each strick in an unique rally
        seq = group[FEATURES].values.tolist()
        strick_count = len(group)
        
        for window_size in range(1, min(strick_count, MAX_SEQ_LEN + 1)):
            for k in range(0, strick_count - window_size):
                hist_seq = seq[k:k+window_size]
                hist_seq = pad_sequences([hist_seq[-MAX_SEQ_LEN:]], maxlen=MAX_SEQ_LEN, 
                                        padding='pre', truncating='pre', value=0)[0]
                # add features (X) and 3 targets (y)
                X.append(hist_seq)
                y_server.append(group.iloc[k+window_size]["serverGetPoint"])
                y_action.append(group.iloc[k+window_size]["actionId"])
                y_point.append(group.iloc[k+window_size]["pointId"])
                 
    return np.array(X), np.array(y_server), np.array(y_action), np.array(y_point)

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
X_train, y_server, y_action, y_point = make_sequences(df_train)
np.save("X.npy", X_train)
np.save("y_server.npy", y_server)
np.save("y_action.npy", y_action)
np.save("y_point.npy", y_point)

print("✅ preprocessed data saved")

# check data length
num_features = len(FEATURES)
num_server_classes = len(np.unique(y_server))
num_action_classes = len(np.unique(y_action))
num_point_classes  = len(np.unique(y_point))

print("category count of server: ",num_server_classes)
print("category count of action: ", num_action_classes)
print("category count of point: ", num_point_classes)
print("length of training sequence data: ", len(X_train))

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

# save models
model_server.save("model_server.keras")
model_action.save("model_action.keras")
model_point.save("model_point.keras")

print("✅ three models saved")

rally_ids = []
pred_server, pred_action, pred_point = [], [], []

for rally_uid, group in df_test.groupby("rally_uid"):
    seq = group[FEATURES].values

    # data preprocessing for testing data: Padding
    seq_padded = pad_sequences([seq], maxlen=MAX_SEQ_LEN, dtype='float32', padding='pre', truncating='pre')

    # make prediction
    ps = model_server.predict(seq_padded)
    pa = model_action.predict(seq_padded)
    pp = model_point.predict(seq_padded)

    # add predictions
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
