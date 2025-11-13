# ================================= libraries =================================
import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Concatenate
from tensorflow.keras.layers import TimeDistributed

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
EPOCHS = 35
EMBED_DIM = 16
VOCAB_SIZE = [3, 5, 5, 4, 5, 7, 11, 20, 5]

df_train = pd.read_csv(TRAIN_FILE)

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

def data_preprocessing(read_data):
    if read_data:
        # modify classes encoding
        df_train["actionId"] = df_train["actionId"].replace(-1, 19)
        df_train["pointId"] = df_train["pointId"].replace(-1, 10)

        # group data with rally_uid and save sequences
        X_train, y_server, y_action, y_point = make_sequences(df_train)
        np.save("X.npy", X_train)
        np.save("y_server.npy", y_server)
        np.save("y_action.npy", y_action)
        np.save("y_point.npy", y_point)
        print("✅ preprocessed data saved")

        return X_train, y_server, y_action, y_point
    else:
        # load exsiting data
        X_train = np.load("data_len_8/X.npy")
        y_server = np.load("data_len_8/y_server.npy")
        y_action = np.load("data_len_8/y_action.npy")
        y_point = np.load("data_len_8/y_point.npy")
        return X_train, y_server, y_action, y_point

def build_multi_embedding_lstm(vocab_sizes, num_server_classes, num_action_classes, num_point_classes, embedding_dim=16):
    inputs, embedded_features = [], []

    for i, feat in enumerate(FEATURES):
        inp = Input(shape=(MAX_SEQ_LEN,), name=feat)
        emb = Embedding(
            input_dim=vocab_sizes[i], output_dim=embedding_dim,
            mask_zero=True, name=f"{feat}_emb"
        )(inp)
        inputs.append(inp)
        embedded_features.append(emb)

    x = Concatenate(name="concat_embeddings")(embedded_features)
    x = LSTM(128, dropout=0.3, recurrent_dropout=0.3, name="shared_lstm")(x)

    # multitask
    out_server = Dense(num_server_classes, activation='softmax', name='serverGetPoint_out')(x)
    out_action = Dense(num_action_classes, activation='softmax', name='actionId_out')(x)
    out_point = Dense(num_point_classes, activation='softmax', name='pointId_out')(x)

    model = Model(inputs=inputs, outputs=[out_server, out_action, out_point], name="MultiFeature_MultiTask_LSTM")

    model.compile(
        optimizer=Adam(1e-3),
        loss={
            "serverGetPoint_out": "sparse_categorical_crossentropy",
            "actionId_out": "sparse_categorical_crossentropy",
            "pointId_out": "sparse_categorical_crossentropy",
        },
        metrics={
            "serverGetPoint_out": ["accuracy"],
            "actionId_out": ["accuracy"],
            "pointId_out": ["accuracy"],
        }
    )
    return model

# ================================= main =================================
read_data = input("read train.csv and do data preprocessing?")
X_train, y_server, y_action, y_point = data_preprocessing(int(read_data))

# load test.csv
df_test = pd.read_csv(TEST_FILE)
df_test["actionId"] = df_test["actionId"].replace(-1, 19)
df_test["pointId"] = df_test["pointId"].replace(-1, 10)

# check data length
num_server_classes = len(np.unique(y_server))
num_action_classes = len(np.unique(y_action))
num_point_classes  = len(np.unique(y_point))

print("category count of server: ",num_server_classes)
print("category count of action: ", num_action_classes)
print("category count of point: ", num_point_classes)
print("length of training sequence data: ", len(X_train))

# separate data
def split_features(X): return [X[:, :, i] for i in range(X.shape[2])]
X_tr, X_val, y_server_tr, y_server_val, y_action_tr, y_action_val, y_point_tr, y_point_val = train_test_split(
    X_train, y_server, y_action, y_point, test_size=0.1, random_state=42)
X_tr_list, X_val_list = split_features(X_tr), split_features(X_val)

# modle settings and training
print("\nTraining multi-embedding LSTM model...")
model = build_multi_embedding_lstm(VOCAB_SIZE, num_server_classes, num_action_classes, num_point_classes, EMBED_DIM)
history = model.fit(
    X_tr_list,
    {"serverGetPoint_out": y_server_tr, "actionId_out": y_action_tr, "pointId_out": y_point_tr},
    epochs=EPOCHS,
    batch_size=128,
    verbose=2,
    validation_data=(X_val_list, {"serverGetPoint_out": y_server_val, "actionId_out": y_action_val, "pointId_out": y_point_val})
)

model.save("model_multitask.keras")
print("✅ Multi-output model saved!")

# predict test.csv
print("\nstart predicting for test.csv ...")
rally_ids = df_test["rally_uid"].unique()
pred_server, pred_action, pred_point = [], [], []

for rally_uid, group in df_test.groupby("rally_uid"):
    seq = group[FEATURES].values
    seq_padded = pad_sequences([seq], maxlen=MAX_SEQ_LEN, padding='pre', truncating='pre', value=0)
    X_test_list = [seq_padded[:, :, i] for i in range(seq_padded.shape[2])]
    ps, pa, pp = model.predict(X_test_list, verbose=2)
    pred_server.append(np.argmax(ps))
    pred_action.append(np.argmax(pa))
    pred_point.append(np.argmax(pp))

print(f"Prediction complete for {len(rally_ids)} rallies")

# check data length align
print("len pred server: ", len(pred_server))
print("len pred action: ", len(pred_action))
print("len pred point: ", len(pred_point))
assert len(pred_server) == len(pred_action) == len(pred_point) == len(rally_ids), \
    f"❌ 預測結果長度不符：rally_ids={len(rally_ids)}, server={len(pred_server)}, action={len(pred_action)}, point={len(pred_point)}"

# write result into submission.csv
df_submission = pd.read_csv(RESULT_FILE)

df_submission["serverGetPoint"] = pred_server
df_submission["actionId"] = pred_action
df_submission["pointId"] = pred_point

df_submission["actionId"] = df_submission["actionId"].replace(19, -1)
df_submission["pointId"] = df_submission["pointId"].replace(10, -1)

df_submission.to_csv(RESULT_FILE, index=False)
print("result saved to ", RESULT_FILE)
