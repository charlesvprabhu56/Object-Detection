# Importing the libraries
import numpy as np
from Evaluation import evaluation
from Model_yolov8_GRU import Model_Yolov8
from keras.models import Sequential
from keras.layers import Dense, Dropout, GRU


def Model_GRU(Image, Target, Epoch=None, sol=None):
    Image = Model_Yolov8(Image, Target)

    if Epoch is None:
        Epoch = [100, 200, 300, 400]
    if sol is None:
        sol = [5, 5, 5, 5]

    learnperc = round(Image.shape[0] * 0.75)  # Split Training and Testing Datas
    X_train = Image[:learnperc, :]
    y_train = Target[:learnperc, :]
    X_test = Image[learnperc:, :]
    Y_test = Target[learnperc:, :]

    Train_Data = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    Test_Data = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))
    # The GRU architecture
    regressorGRU = Sequential()
    # First GRU layer with Dropout regularisation
    regressorGRU.add(GRU(units=10, return_sequences=True, input_shape=(1, Train_Data.shape[2]), activation='tanh'))
    regressorGRU.add(Dropout(0.2))
    # The output layer
    regressorGRU.add(Dense(y_train.shape[1]))
    # Compiling the RNN
    regressorGRU.compile(optimizer='adam',
                         loss='mean_squared_error')
    Train_Data = np.asarray(Train_Data).astype(np.float32)
    TestX = np.asarray(Test_Data).astype(np.float32)
    regressorGRU.fit(Train_Data, y_train, epochs=sol[3])
    pred = regressorGRU.predict(TestX)
    pred = np.reshape(pred, (pred.shape[0], pred.shape[2]))
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    Eval = evaluation(pred, Y_test)
    return Eval, pred
