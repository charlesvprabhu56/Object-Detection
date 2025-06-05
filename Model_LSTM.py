import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM
from Evaluation import evaluation


# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def LSTM_train(trainX, trainY, testX, testY, epoch):
    batch_size = 4
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    model = Sequential()
    model.add(LSTM(64, input_shape=(1, trainX.shape[2])))
    model.add(Dense(trainY.shape[1]))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    trainX = np.asarray(trainX).astype(np.float32)
    testX = np.asarray(testX).astype(np.float32)
    model.fit(trainX, trainY, epochs=epoch)
    pred = model.predict(testX)
    return pred, model


def Model_LSTM(train_data, train_target, test_data, test_target, epoch):
    out, model = LSTM_train(train_data, train_target, test_data, test_target, epoch)
    pred = np.asarray(out)
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, test_target)

    return Eval, pred

