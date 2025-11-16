## Competition Link
https://www.kaggle.com/competitions/introduction-to-data-secience-ttmatch/overview

## Tasks Description
1. Next stroke (n-th shot) action type prediction: `actionId`
2. Next stroke (n-th shot) landing location prediction: `pointId`
3. Current rally outcome prediction: `serverGetPoint`

## Environment Settings
1. create a virtual environment with **Conda**
2. install required packages (tensorflow...)

## Data Preprocessing: `data_preprocessing.py`
main idea: in each `rally_uid`, use features of previous `MAX_SEQ_LEN` stricks to predict `actionId`,`pointId` and `serverGetPoint` of next strick. `MAX_SEQ_LEN` is set to 8, if the `strickNumber` is less than 8, use `strickNumber` stricks to predict

**parameters**
1. set training data file name
2. select 9 features, `"serveId", "serveNumber", "strickId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"`
3. set max sequence length

**make_sequences_1(df)**
1. group data with `"rally_uid"`
2. in each `"rally_uid"`, make sequence for features data of previous `k` stricks, where `k` is set from 1 to length of that rally. if `k` is less than `MAX_SEQ_LEN`, we need to pad blank data with `0` to make sure that all sequences are in the same length(structure), else if `k` is larger than `MAX_SEQ_LEN`, we only keep `MAX_SEQ_LEN` rows of features data
3. in each iteration, append preprocessec data to a list, and return it at the end, making sure that you turn them into numpy array

**make_sequences_2(df)**
1. group data with `"rally_uid"`
2. run a nested loop for each `"rally_uid"`. outer loop: window size set from 1 to `min(strick_count, 'MAX_SEQ_KEN'+1)`, making sure that window size won't exceed `strick_count` of that rally
3. inner loop: index k set from 1 to `strick_count - window_size`, so we can slide the window through data of that rally and collect history features to make sequences
4. if `window_size` is smaller than `MAX_SEQ_LEN`, we need to pad blank data with `0` to make sure that all sequences are in the same length(structure)
5. in each iteration, append preprocessec data to a list, and return it at the end, making sure that you turn them into numpy array

**def data_preprocessing(file_path)**
1. read `train.csv`
2. replace `-1` with another integer in `"pointId"` and `"actionId"`, because predicting target cannot be negative category
3. call `make_sequences`
4. save preprocessed training data and predicting target into numpy array for latet use

**main**
1. create folder for saving preprocessed data
2. call function `data_preprocessing`

## Model Training `train_model.py`
main idea: use **LSTM**, a model usually used for predicting stock price

**parameters**
1. set input and output file names
2. select 9 features, `"serveId", "serveNumber", "strickId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"`
3. set hyperparameters for model, including `EPOCHS`, `BATCH`, `EMBED_DIMS`
4. `VOCAB_SIZE`  is number of category of each features plus `0`, which means no data

**def build_multi_embedding_lstm**
1. embedding featrues one by one, then concatenate them
2. set a one layerLSTM model
3. model gives multi-task outputs (three different target)
4. use `sparse_categorical_crossentropy` for integer label

**main**
1. load preprocessed data
2. split_features for separating data, then we can embedding each feature one by one
3. call `build_multi_embedding_lstm` and train model
4. read `test.csv`, do the same replacement on `"actionId"` and `"pointId"`
5. group testing data with `rally_uid`, padding sequences for them, just like what we did in data preprocessing stage
6. feed features data to the model, making predictions and saving results
7. check if length of results equals to rallys in the testing data file
8. write results into csv, noted that we have to 
undo the replacement on -1

## How to execute?
place `train.csv`, `test.csv`, `data_preprocessing.py` and `train_model.py` under the same folder, then run instructions below:
```
conda activate <virtual environment name>
python data_preprocessing.py
python train_model.py
```

## References
1.  https://medium.com/data-scientists-playground/lstm-%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92-%E8%82%A1%E5%83%B9%E9%A0%90%E6%B8%AC-cd72af64413a
2. https://claire-chang.com/2023/01/07/%E4%BA%A4%E5%8F%89%E7%86%B5%E7%9B%B8%E9%97%9C%E6%90%8D%E5%A4%B1%E5%87%BD%E6%95%B8%E7%9A%84%E6%AF%94%E8%BC%83/