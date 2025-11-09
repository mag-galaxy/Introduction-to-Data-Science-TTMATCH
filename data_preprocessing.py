import numpy as np
import pandas as pd

# from tensorflow.keras.preprocessing.sequence import pad_sequences

TRAIN_FILE = "train.csv"
TARGET = ["serverGetPoint", "actionId", "pointId"]
FEAT_NOT_USED = ["rally_uid", "sex", "match", "numberGame", "rally_id", "strickNumber", "let"]
FEATURES = [
    "serveId", "serveNumber", "strickId", "handId", 
    "strengthId", "spinId", "pointId", "actionId", "positionId"
]
MAX_SEQ_LEN = 6

# def make_sequences(df):
#     X, y_server, y_action, y_point = [], [], [], []
#     for rally_uid, group in df.groupby("rally_uid"):
#         # seq is features of each strick in an unique rally
#         seq = group[FEATURES].values.tolist()
#         strick_count = len(group)
        
#         # 0~(k-1)th -> k-th
#         for window_size in range(1, min(strick_count, MAX_SEQ_LEN + 1)):
#             for k in range(0, strick_count - window_size):
#                 hist_seq = seq[k:k+window_size]
#                 hist_seq = pad_sequences([hist_seq[-MAX_SEQ_LEN:]], maxlen=MAX_SEQ_LEN, 
#                                         padding='pre', truncating='pre', value=0)[0]
#                 # add features (X) and 3 targets (y)
#                 print(f"rally_uid = {rally_uid}, strickCount = {strick_count}, window_size = {window_size}, k = {k},\nhist_seq =\n{hist_seq}")
#                 print("y_server", group.iloc[k+window_size]["serverGetPoint"])
#                 print("y_action", group.iloc[k+window_size]["actionId"])
#                 print("y_point", group.iloc[k+window_size]["pointId"])
#                 X.append(hist_seq)
#                 y_server.append(group.iloc[k+window_size]["serverGetPoint"])
#                 y_action.append(group.iloc[k+window_size]["actionId"])
#                 y_point.append(group.iloc[k+window_size]["pointId"])
                 
#     return np.array(X), np.array(y_server), np.array(y_action), np.array(y_point)
    
# main
print("read data...")
# df_train = pd.read_csv(TRAIN_FILE)

# X_train, y_server, y_action, y_point = make_sequences(df_train)

# np.save("X.npy", X_train)
# np.save("y_server.npy", y_server)
# np.save("y_action.npy", y_action)
# np.save("y_point.npy", y_point)

# print("len of sequence data: ",len(X_train))
# print("✅ preprocessed data saved")

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