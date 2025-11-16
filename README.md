## Competition Link
https://www.kaggle.com/competitions/introduction-to-data-secience-ttmatch/overview

## Tasks Description
1. Next stroke (n-th shot) action type prediction: `actionId`
2. Next stroke (n-th shot) landing location prediction: `pointId`
3. Current rally outcome prediction: `serverGetPoint`

## Environment Settings
1. create a virtual environment with **Conda**
2. install required packages (tensorflow...)

:::
Note: best score ver. only
:::

## Data Preprocessing: `data+_preprocessing.py`
main idea: in each `rally_uid`, use features of previous n stricks to predict `actionId`,`pointId` and `serverGetPoint` of next strick, n is set from 1 to `MAX_SEQ_LEN`

**parameters**
1. set training data file name
2. select 9 features, `"serveId", "serveNumber", "strickId", "handId", "strengthId", "spinId", "pointId", "actionId", "positionId"`
3. set max sequence length

**make_sequences(df)**
1. group data with `"rally_uid"`
2. outer loop: window size set from 1 to `min(strick_count, 'MAX_SEQ_KEN'+1)`, making sure that window size won't exceed `strick_count` of that rally
3. inner loop: index k set from 1 to `strick_count - window_size`, so we can slide the window through data of that rally and collect history features to make sequences
4. if `window_size` is smaller than `MAX_SEQ_LEN`, we need to pad blank data with `0` to make sure that all sequences are in the same length(structure)
5. in each iteration, append preprocessec data to a list, and return it at the end

**def data_preprocessing(file_path)**
1. read `train.csv`
2. replace `-1` with another integer in `"pointId"` and `"actionId"`, because predicting target cannot be negative category
3. call `make_sequences`
4. save preprocessed training data and predicting target into numpy array for latet use

**main**
1. create folder for saving preprocessed data
2. call function `data_preprocessing`

## Model Training

## Making Predictions
1. read `test.csv`, do the same replacement on `"actionId"` and `"pointId"`
2. group testing data with `rally_uid`, padding sequences for them, just like what we do in data preprocessing stage
3. feed features data to the model, making predictions and saving results
4. check if length of results equals to rallys in the testing data file
5. write results into csv, noted that we have to 
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