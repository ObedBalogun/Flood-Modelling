import math

import tensorflow as tf
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization, LSTMCell
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from collections import deque
import random
import time

SEQUENCE_OF_DAYS = 1
EPOCHS = 50
BATCH_SIZE = 32



def preprocess_df(df):
    df = df.drop('Future Runoff Values', 1)
    for col in df.columns:
        if col not in ['Target', 'Year', 'Month', 'Day', 'Flood Event','Precipitation']:
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
    # divide dataset by classification
    flood = []
    non_flood = []
    for seq, target in sequential_data:
        if target == 0:
            non_flood.append([seq, target])
        elif target == 1:
            flood.append([seq, target])
    print(len(flood), len(non_flood))
    random.shuffle(flood)
    random.shuffle(non_flood)
    print(len(flood),len(non_flood))
    lower = min(len(flood), len(non_flood))

    # Balance Dataset
    flood = flood[:lower]
    non_flood = non_flood[:lower]


    sequential_data = flood + non_flood

    random.shuffle(sequential_data)

    X, y = [], []

    for seq, target in sequential_data:
        X.append(seq)
        y.append(target)

    return np.array(X), np.array(y)


def classify(value):
    if value == 'flood':
        return 1
    else:
        return 0

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


locations = ['Ibadan', 'Ikeja', 'Osogbo', 'Abeokuta']

for location in locations:
    NAME = f"Flood suceptibility model for {location}"
    dataset = pd.ExcelFile(f'C:/Users/Obed/Documents/MastersProject/data/runoff_data/Flood Events/2009-2018/{location} Flood Events.xlsx')
    dataset = dataset.parse('Sheet1')
    dataset = pd.DataFrame(dataset)
    dataset['Future Runoff Values'] = dataset['Total Runoff'].shift(-3)
    dataset['Target'] = list(map(classify, dataset['Flood Event']))

    times = sorted(dataset.index.values)
    last_20pct = times[-int(0.2 * len(times))]
    validation_dataset = dataset[(dataset.index >= last_20pct)]
    dataset = dataset[(dataset.index < last_20pct)]

    train_x, train_y = preprocess_df(dataset)
    validation_x, validation_y = preprocess_df(validation_dataset)

    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    validation_x = validation_x.reshape((validation_x.shape[0], 1, validation_x.shape[1]))

    print(f'train_data: {len(train_x)} validation:{len(validation_x)}')
    # print(f'Non Flood:{train_y.count(0)}, flood:{train_y.count(1)}')
    # print(f'Validation Non Flood:{validation_y.count(0)}, flood:{validation_y.count(1)}')


    model = Sequential()
    # input layer
    model.add(LSTM(128, input_shape=(train_x.shape[1], train_x.shape[2]),return_sequences=True))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(64, activation='relu'))

    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Dense(1))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, decay=1e-6)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['accuracy','mse',f1_m,precision_m, recall_m])
    # model.summary()

    tensorboard = TensorBoard(log_dir=f'logs/{NAME}')

    filepath = "RNN_Final-{epoch:02d}-{val_accuracy:.3f}"

    checkpoint = ModelCheckpoint("models/{}.model".format(filepath, monitor='val_accuracy', verbose=1,save_best_only =True,mode='max'))

    history = model.fit(train_x, train_y,
                        batch_size=BATCH_SIZE,
                        epochs=EPOCHS,
                        validation_data=(validation_x, validation_y),
                        callbacks=[tensorboard, checkpoint])

    # make predictions
    trainPredict = model.predict(train_x)
    testPredict = model.predict(validation_x)


    # Score model
    score = model.evaluate(validation_x, validation_y, verbose=0)
    # score = model.evaluate(validation_x, validation_y, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print('Mean Square Error:', score[2])
    print('F1 score:', score[3])
    print('precision:', score[4])
    print('recall:', score[5])
    # save model
    # model.save("models/{}".format(NAME))



# evaluate the model
# loss, accuracy, f1_score, precision, recall = model.evaluate(Xtest, ytest, verbose=0)