import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import deque
import random
import time

SEQUENCE_OF_DAYS = 1
EPOCHS = 20
BATCH_SIZE = 32
NAME = f"Flood suceptibility model for ib"


def preprocess_df(df):
    df = df.drop('Future Runoff Values', 1)

    for col in df.columns:
        if col not in ['Target', 'Year', 'Month', 'Day', 'Flood Event']:
            df[col] = df[col].pct_change()
            df[col] = df[col].replace(np.inf, 0)
            df[col] = df[col].replace(np.nan, 0)
            df[col] = preprocessing.scale(df[col].values)

    sequential_data = []
    prev_days = deque(maxlen=SEQUENCE_OF_DAYS)
    for i in df.values:
        prev_days.append(i[4])
        if len(prev_days) == SEQUENCE_OF_DAYS:
            sequential_data.append([np.array(prev_days), i[-1]])
    random.shuffle(sequential_data)

    flood = []
    non_flood = []
    for seq, target in sequential_data:
        if target == 0:
            non_flood.append([seq, target])
        elif target == 1:
            flood.append([seq, target])

    random.shuffle(flood)
    random.shuffle(non_flood)
    print(len(flood), len(non_flood))

    lower = min(len(flood), len(non_flood))

    flood = flood[:lower]
    non_flood = non_flood[:lower]

    sequential_data = flood + non_flood

    random.shuffle(sequential_data)

    X, y = [], []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)
    print( X,y)
    return np.array(X), np.array(y)


def classify(value):
    if value == 'flood':
        return 1
    else:
        return 0


dataset = pd.ExcelFile(f'C:/Users/Obed/Documents/MastersProject/data/runoff_data/Flood Events/Ibadann Flood Events.xlsx')
dataset = dataset.parse('Sheet1')
dataset = pd.DataFrame(dataset)
dataset['Future Runoff Values'] = dataset['Total Runoff'].shift(-3)
dataset['Target'] = list(map(classify, dataset['Flood Event']))

# print(dataset[['Total Runoff','Flood Event','Target']].head())
times = sorted(dataset.index.values)
last_5pct = times[-int(0.2 * len(times))]
validation_dataset = dataset[(dataset.index >= last_5pct)]
dataset = dataset[(dataset.index < last_5pct)]

train_x, train_y = preprocess_df(dataset)
validation_x, validation_y = preprocess_df(validation_dataset)

train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
validation_x = validation_x.reshape((validation_x.shape[0], 1, validation_x.shape[1]))

print(f'train_data: {len(train_x)} validation:{len(validation_x)}')
# print(f'Non Flood:{train_y.count(0)}, flood:{train_y.count(1)}')
# print(f'Validation Non Flood:{validation_y.count(0)}, flood:{validation_y.count(1)}')


model = Sequential()
model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
# model.add(LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.2))
model.add(BatchNormalization())
#
model = Sequential()
# model.add(LSTM(128, input_shape=(train_x.shape[1:]), return_sequences=True))
model.add(LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.1))
model.add(BatchNormalization())

model = Sequential()
# model.add(LSTM(128, input_shape=(train_x.shape[1:])))
model.add(LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2])))
model.add(Dropout(0.2))
model.add(BatchNormalization())

model.add(Dense(32, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(2, activation="softmax"))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

model.compile(loss='mse',
              optimizer=optimizer,
              metrics=['accuracy'])
tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"

checkpoint = ModelCheckpoint(
    "models/{}.model".format(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max'))

history = model.fit(train_x, train_y,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    validation_data=(validation_x, validation_y),
                    callbacks=[tensorboard, checkpoint])

score = model.evaluate(validation_x, validation_y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




