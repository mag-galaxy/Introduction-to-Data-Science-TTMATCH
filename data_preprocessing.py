# ================================= libraries =================================
import numpy as np
import pandas as pd
import os
import argparse

from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================================= arguments =================================
parser = argparse.ArgumentParser(description="all in one")
parser.add_argument('-l', '--sequence_len', default=8, help="sequence len")
args = parser.parse_args()

# ================================= parameters =================================
TRAIN_FILE = "train.csv"
FEATURES = [
    "serveId", "serveNumber", "strickId", "handId", 
    "strengthId", "spinId", "pointId", "actionId", "positionId"
]
MAX_SEQ_LEN = int(args.sequence_len)
FOLDER_NAME = f"data_len_{MAX_SEQ_LEN}"

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

def data_preprocessing(file_path):
    df_train = pd.read_csv(file_path)

    # modify classes encoding
    df_train["actionId"] = df_train["actionId"].replace(-1, 19)
    df_train["pointId"] = df_train["pointId"].replace(-1, 10)

    # group data with rally_uid and save sequences
    X_train, y_server, y_action, y_point = make_sequences(df_train)
    np.save(f"{FOLDER_NAME}/X.npy", X_train)
    np.save(f"{FOLDER_NAME}/y_server.npy", y_server)
    np.save(f"{FOLDER_NAME}/y_action.npy", y_action)
    np.save(f"{FOLDER_NAME}/y_point.npy", y_point)
    print("✅ preprocessed data saved")

    return

# ================================= main =================================
print("create new folder...")
os.makedirs(FOLDER_NAME, exist_ok=True)
print("start data preprocessing...")
data_preprocessing(TRAIN_FILE)