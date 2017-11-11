from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import Activation
from sklearn import preprocessing

import pandas
import numpy
import matplotlib.pyplot as plt
from sys import exit

def load_data():
    df = [pandas.read_csv("data/indices_I101_1d_" + str(year)+ ".csv", usecols=[0,4]) # 0 = 日付, 4 = 終値
        for year in range(2007, 2017)]
    df = pandas.concat(df)

    df.columns = ["date", "close"]
    df["date"] = pandas.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.sort_values(by="date")
    df = df.reset_index(drop=True)
    df["close"] = preprocessing.scale(df["close"])

    return df[["close"]]

def create_dataset(df, look_back=10, predict_day=1):
    # look_back 日分のデータから次の日の終値を予測
    # predict_day 日後の株価を予測
    X, Y = [], []
    for i in range(len(df) - look_back - predict_day):
        X.append(df.iloc[i:(i+look_back)].as_matrix())
        Y.append(df.iloc[i+look_back+predict_day].as_matrix())

    return numpy.array(X), numpy.array(Y)

def plot_history_loss(fit):
    # Plot the loss in the history
    axL.plot(fit.history['loss'],label="loss for training")
    axL.set_title('train loss (mae)')
    axL.set_xlabel('epoch')
    axL.set_ylabel('loss')
    axL.legend(loc='upper right')

    axR.plot(fit.history['val_loss'],label="loss for validation")
    axR.set_title('test loss (mae)')
    axR.set_xlabel('epoch')
    axR.set_ylabel('loss')
    axR.legend(loc='upper right')

if __name__ == "__main__":
    #pandas.set_option("display.max_rows", 1000)
    data = load_data()

    split_pos = int(len(data) * 0.8)
    X_train, Y_train = create_dataset(data.iloc[0:split_pos], predict_day=10)
    X_test,  Y_test  = create_dataset(data.iloc[split_pos:])

    model = Sequential()
    model.add(LSTM(50,
                   batch_input_shape=(None, 10, 1),
                   return_sequences=False))
    model.add(Dense(1))
    model.add(Activation("linear"))
    model.compile(loss="mae", optimizer="adam")

    fit = model.fit(X_train, Y_train,
                    batch_size=50,
                    epochs=100,
                    validation_data=(X_test, Y_test), 
                    verbose=1)

    fig, (axL, axR) = plt.subplots(ncols=2, figsize=(10, 4))
    plot_history_loss(fit)

    predicted = model.predict(X_test)
    result = pandas.DataFrame(predicted)
    result["actual"] = Y_test
    result.plot()
    plt.show()

