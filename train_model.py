import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ---------- 1. 讀取資料 ----------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# ---------- 2. 資料前處理 ----------
# 假設 train 裡有：rally_uid, shot_id, features..., is_end_shot (-1表示結束拍)
# 我們先把資料依 rally_uid 分組
train_groups = train.groupby("rally_uid")

X, y = [], []

for _, df in train_groups:
    df = df.sort_values("shot_id")
    for k in range(1, len(df)):
        # 取前 k-1 拍作為輸入，預測第 k 拍是否為結束拍
        seq = df.iloc[:k][["feature1", "feature2", "feature3", ...]].values
        label = (df.iloc[k]["is_end_shot"] == -1).astype(int)
        X.append(seq)
        y.append(label)

# 序列填補（不同長度）
X_padded = pad_sequences(X, padding='post', dtype='float32')
y = np.array(y)

# ---------- 3. 分割資料 ----------
X_train, X_val, y_train, y_val = train_test_split(X_padded, y, test_size=0.1, random_state=42)

# ---------- 4. 建立模型 ----------
model = Sequential([
    Masking(mask_value=0.0, input_shape=(X_padded.shape[1], X_padded.shape[2])),
    LSTM(128, return_sequences=False),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# ---------- 5. 訓練 ----------
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, batch_size=128)

# ---------- 6. 處理 test.csv ----------
# test.csv 裡每個 rally_uid 只提供前 n-1 拍，我們要預測第 n 拍是否為結束拍
test_groups = test.groupby("rally_uid")
X_test = []

for _, df in test_groups:
    df = df.sort_values("shot_id")
    seq = df[["feature1", "feature2", "feature3", ...]].values
    X_test.append(seq)

X_test_padded = pad_sequences(X_test, padding='post', maxlen=X_padded.shape[1], dtype='float32')

# ---------- 7. 預測 ----------
preds = model.predict(X_test_padded)
preds_label = (preds > 0.5).astype(int)

# ---------- 8. 輸出結果 ----------
submission = pd.DataFrame({
    "rally_uid": test["rally_uid"].unique(),
    "is_end_shot_pred": preds_label.flatten()
})
submission.to_csv("submission.csv", index=False)
