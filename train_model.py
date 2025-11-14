# ================================= libraries =================================
import numpy as np
import pandas as pd
import argparse

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Masking
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, Concatenate
from tensorflow.keras.layers import TimeDistributed

# ================================= arguments =================================
parser = argparse.ArgumentParser(description="all in one")
parser.add_argument('-l', '--sequence_len', default=8, help="sequence len")
args = parser.parse_args()

# ================================= parameters =================================
TEST_FILE = "test.csv"
RESULT_FILE = "111504504_submission.csv"
FEATURES = [
    "serveId", "serveNumber", "strickId", "handId", 
    "strengthId", "spinId", "pointId", "actionId", "positionId"
]
MAX_SEQ_LEN = int(args.sequence_len)
FOLDER_NAME = f"data_len_{MAX_SEQ_LEN}"
EPOCHS = 40
EMBED_DIM = 16
BATCH = 64
VOCAB_SIZE = [3, 5, 5, 4, 5, 7, 11, 20, 5]

# ================================= functions =================================
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
    x = LSTM(128, dropout=0.3, recurrent_dropout=0.3, name="shared_lstm", return_sequences=True)(x)
    x = LSTM(64)(x)

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
# load preprocessed data
print("load preprocessed data...")
X_train = np.load(f"{FOLDER_NAME}/X.npy")
y_server = np.load(f"{FOLDER_NAME}/y_server.npy")
y_action = np.load(f"{FOLDER_NAME}/y_action.npy")
y_point = np.load(f"{FOLDER_NAME}/y_point.npy")

num_server_classes = len(np.unique(y_server))
num_action_classes = len(np.unique(y_action))
num_point_classes  = len(np.unique(y_point))

# load test.csv
print("load test.csv")
df_test = pd.read_csv(TEST_FILE)
df_test["actionId"] = df_test["actionId"].replace(-1, 19)
df_test["pointId"] = df_test["pointId"].replace(-1, 10)

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
    batch_size=BATCH,
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
