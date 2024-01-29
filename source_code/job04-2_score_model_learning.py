import numpy as np
import matplotlib.pyplot as plt
import os

from tensorflow.keras.models import *
from tensorflow.keras.layers import *

file = "../learning_data/score_data_max_199_wordsize_24734.npy"

max = int(os.path.splitext(file)[0].split("/")[-1].split("_")[3])  # 파일명에서 문장의 최대 거리를 가져옴
wordsize = int(os.path.splitext(file)[0].split("/")[-1].split("_")[5])  # 파일명을 이용하여 단어의 개수를 가져옴

print(max, wordsize)

X_train, X_test, Y_train, Y_test = np.load(file, allow_pickle=True)  # 데이터를 불러옴

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential(
    [
        Embedding(wordsize, wordsize // 40, input_length=max),
        Conv1D(40, kernel_size=5, padding="same", activation="relu"),
        MaxPooling1D(pool_size=1),
        LSTM(wordsize // 1200, activation="tanh", return_sequences=True),
        Dropout(0.3),
        LSTM(wordsize // 2400, activation="tanh", return_sequences=True),
        Dropout(0.3),
        LSTM(wordsize // 2400, activation="tanh", return_sequences=True),
        Dropout(0.3),
        LSTM(wordsize // 2400, activation="tanh"),
        Dropout(0.3),
        Flatten(),
        Dense(wordsize // 1200, activation="relu"),
        Dropout(0.3),
        Dense(wordsize // 1200, activation="relu"),
        Dense(wordsize // 600, activation="relu"),
        Dense(11, activation="softmax")
    ]
)

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

fit_hist = model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_test, Y_test))

model.save("../models/score_category_classification_model_{}_{}_{}.h5".format(max, wordsize, fit_hist.history["val_accuracy"][-1]))

plt.plot(fit_hist.history["val_accuracy"], label="validation accuracy")
plt.plot(fit_hist.history["accuracy"], label="accuracy")

plt.legend()
plt.show()
plt.savefig("../image/score_accuracy.png")
