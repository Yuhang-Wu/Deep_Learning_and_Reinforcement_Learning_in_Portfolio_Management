import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

CONST_TRAINING_SEQUENCE_LENGTH = 12
CONST_TESTING_CASES = 5


def dataNormalization(data):
    return [(datum - data[0]) / data[0] for datum in data]


def dataDenormalization(data, base):
    return [(datum + 1) * base for datum in data]


def getDeepLearningData(ticker):

    # Step 1. Load Data
    data = pd.read_csv("../Data/IntradayCN/" +
                       ticker + '.csv')['close'].tolist()

    # Step 2. Building Training data
    dataTraining = []
    for i in range(len(data) - CONST_TESTING_CASES * CONST_TRAINING_SEQUENCE_LENGTH):
        dataSegment = data[i: i + CONST_TRAINING_SEQUENCE_LENGTH + 1]
        dataTraining.append(dataNormalization(dataSegment))

    dataTraining = np.array(dataTraining)
    # np.random.shuffle(dataTraining) # order matter
    X_Training = dataTraining[:, :-1]
    Y_Training = dataTraining[:, -1]

    X_Training.shape
    Y_Training.shape

    # Step 3. Build Test data
    X_Testing = []
    Y_Testing_Base = []
    for i in range(CONST_TESTING_CASES, 0, -1):
        dataSegment = data[-(i + 1) * CONST_TRAINING_SEQUENCE_LENGTH:
                           - i * CONST_TRAINING_SEQUENCE_LENGTH]
        Y_Testing_Base.append(dataSegment[0])
        X_Testing.append(dataNormalization(dataSegment))

    Y_Testing = data[-CONST_TESTING_CASES * CONST_TRAINING_SEQUENCE_LENGTH:]

    X_Testing = np.array(X_Testing)
    Y_Testing = np.array(Y_Testing)

    X_Testing.shape
    Y_Testing.shape

    # Step 4. Reshape for deep learning
    X_Training = np.reshape(
        X_Training, (X_Training.shape[0], X_Training.shape[1], 1))
    X_Testing = np.reshape(
        X_Testing, (X_Testing.shape[0], X_Testing.shape[1], 1))

    X_Training.shape
    X_Testing.shape

    return X_Training, Y_Training, X_Testing, Y_Testing, Y_Testing_Base


def predict(model, X):
    predictionNormalized = []

    for i in range(len(X)):
        data = X[i]
        result = []

        for j in range(CONST_TRAINING_SEQUENCE_LENGTH):
            predicted = model.predict(data[np.newaxis, :, :])[0, 0]
            result.append(predicted)
            data = data[1:]  # kick out the first element
            data = np.insert(
                data, [CONST_TRAINING_SEQUENCE_LENGTH - 1], predicted, axis=0)

        predictionNormalized.append(result)

    return predictionNormalized


def plotResult(Y_hat, Y):
    plt.plot(Y)

    for i in range(len(Y_hat)):
        padding = [None for _ in range(i * CONST_TRAINING_SEQUENCE_LENGTH)]
        plt.plot(padding + Y_hat[i])

    plt.show()


def predictLSTM(ticker):
    # Step 1. Load data
    X_Training, Y_Training, X_Testing, Y_Testing, Y_Testing_Base = getDeepLearningData(
        ticker="000001")

    # Step 2. Build model
    model = Sequential()

    model.add(LSTM(input_dim=1,
                   output_dim=50,
                   return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(output_dim=500,
                   return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(output_dim=1))
    model.add(Activation('linear'))

    model.compile(loss='mse', optimizer='rmsprop')

    # Step 3. Train model
    model.fit(X_Training, Y_Training, batch_size=512,
              epochs=10, validation_split=0.05)

    # Step 4. Predict
    predictionNormalized = predict(model, X_Testing)

    # Step 5. De-nomalize
    predictions = []
    for i, row in enumerate(predictionNormalized):
        predictions.append(dataDenormalization(row, Y_Testing_Base[i]))

    # Step 6. Plot
    plotResult(predictions, Y_Testing)
    predictions, Y_Testing
