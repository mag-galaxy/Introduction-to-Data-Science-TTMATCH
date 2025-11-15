# ================================= libraries =================================
import numpy as np
import pandas as pd
import argparse

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization, Input, Embedding, Concatenate, Lambda
from tensorflow.keras.models import Model, Sequential 
from tensorflow.keras.optimizers import Adam

# ================================= parameters =================================
TEST_FILE = "test.csv"
RESULT_FILE = "111504504_submission.csv"
FEATURES = [
    "serveId", "serveNumber", "strickId", "handId", 
    "strengthId", "spinId", "pointId", "actionId", "positionId"
]
MAX_SEQ_LEN = 8
FOLDER_NAME = f"data_len_{MAX_SEQ_LEN}"
EPOCHS = 30
BATCH = 64
VOCAB_SIZE = [3, 5, 5, 4, 5, 7, 11, 20, 5]

# ------------------------------------------------------------
# 1. 自動依 vocab size 設定 embedding dimension
# ------------------------------------------------------------
def choose_embed_dim(vocab):
    if vocab <= 4:
        return 4
    elif vocab <= 8:
        return 6
    elif vocab <= 15:
        return 8
    else:
        return 12

# ------------------------------------------------------------
# 2. Transformer Encoder Block
# ------------------------------------------------------------
def transformer_block(x, head_size=32, ff_dim=64, num_heads=2, dropout=0.1):
    # Multi-head attention
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=head_size // num_heads, dropout=dropout)(x, x)

    # Add & Norm
    x = LayerNormalization(epsilon=1e-6)(x + attn)

    # Feed-forward
    ffn = Sequential([
        Dense(ff_dim, activation="relu"),
        Dense(head_size)
    ])
    ffn_out = ffn(x)

    # Add & Norm
    x = LayerNormalization(epsilon=1e-6)(x + ffn_out)

    return x

# ------------------------------------------------------------
# 3. 建立 Transformer 多任務模型
# ------------------------------------------------------------
def build_transformer_multitask_model(vocab_sizes, num_server_classes, num_action_classes, num_point_classes, max_seq_len):
    # === 9 個 Input（每個特徵一個） ===
    inputs = [
        Input(shape=(max_seq_len,), name=f"feat_{i}")
        for i in range(len(vocab_sizes))
    ]

    # === 9 個 embedding layers ===
    embedded_features = []
    for i, vocab in enumerate(vocab_sizes):
        embed_dim = choose_embed_dim(vocab)
        emb = Embedding(
            input_dim=vocab,
            output_dim=embed_dim,
            mask_zero=True,       # padding=0 → mask 使用
            name=f"embed_{i}"
        )(inputs[i])
        embedded_features.append(emb)

    # === concat embeddings (批次, 時間, embedding總長度) ===
    x = Concatenate(axis=-1)(embedded_features)

    # === Transformer Encoder (你可增加層數) ===
    x = transformer_block(x, head_size=58, ff_dim=128, num_heads=2)

    # === (可選) 再加第二層 Transformer
    # x = transformer_block(x, head_size=32, ff_dim=64, num_heads=2)

    # === 決策：序列 → 一個向量（用最後 time step 即下一球預測） ===
    x = Lambda(lambda t: t[:, -1, :])(x)

    # === 三個輸出 head ===
    out_server = Dense(num_server_classes, activation='softmax', name='serverGetPoint')(x)
    out_action = Dense(num_action_classes, activation='softmax', name='actionId')(x)
    out_point  = Dense(num_point_classes, activation='softmax', name='pointId')(x)

    # === 建構 Model ===
    model = Model(inputs=inputs, outputs=[out_server, out_action, out_point])

    model.compile(
        optimizer=Adam(1e-3),
        loss={
            "serverGetPoint": "sparse_categorical_crossentropy",
            "actionId": "sparse_categorical_crossentropy",
            "pointId": "sparse_categorical_crossentropy",
        },
        metrics={
            "serverGetPoint": ["accuracy"],
            "actionId": ["accuracy"],
            "pointId": ["accuracy"],
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
print("\nTraining transformer...")
model = build_transformer_multitask_model(
    vocab_sizes=VOCAB_SIZE,
    num_server_classes=num_server_classes,
    num_action_classes=num_action_classes,
    num_point_classes=num_point_classes,
    max_seq_len=MAX_SEQ_LEN
)

model.fit(
    X_tr_list,
    [y_server_tr, y_action_tr, y_point_tr],
    validation_data=(X_val_list, [y_server_val, y_action_val, y_point_val]),
    epochs=EPOCHS,
    batch_size=64,
    verbose=2
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

# write result into submission.csv
df_submission = pd.read_csv(RESULT_FILE)

df_submission["serverGetPoint"] = pred_server
df_submission["actionId"] = pred_action
df_submission["pointId"] = pred_point

df_submission["actionId"] = df_submission["actionId"].replace(19, -1)
df_submission["pointId"] = df_submission["pointId"].replace(10, -1)

df_submission.to_csv(RESULT_FILE, index=False)
print("result saved to ", RESULT_FILE)