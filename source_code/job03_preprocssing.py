import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from konlpy.tag import Okt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pickle


df = pd.read_csv("../learning_data/concated_file.csv")

X = df["RawText"]
Y = df["Polarity"]

label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform(Y)

label = label_encoder.classes_

print(label)

with open("../models/label_encoder.pickle", "wb") as file:
    pickle.dump(label_encoder, file)

onehot_y = to_categorical(labeled_y)

okt = Okt()

for i in range(len(X)):
    if i % 100 == 0:
        print(f"\r형태소 분리 중 : {i/len(X) * 100:.2f}%, {i}/{len(X)}", end="")  # 진행 상황 표시
    X[i] = okt.morphs(X[i], stem=True)
    
print(f"\r형태소 분리 중 : 100.00%")

stopwords = pd.read_csv("../stopwords.csv", index_col=0)

for i in range(len(X)):
    if i % 100 == 0:
        print(f"\r문자열 필터링 중 : {i/len(X) * 100:.2f}%, {i}/{len(X)}", end="")
    words = []
    for j in range(len(X[i])):
        if len(X[i][j]) > 1:
            if X[i][j] not in list(stopwords):
                words.append(X[i][j])

    X[i] = " ".join(words)

print(f"\r문자열 필터링 중 : 100.00%")

token = Tokenizer()
token.fit_on_texts(X)

tokened_x = token.texts_to_sequences(X)
wordsize = len(token.word_index) + 1

print()
print(f"word size : {wordsize}")

with open("../models/word_token.pickle", "wb") as file:
    pickle.dump(token, file)

max = 0

for i in range(len(tokened_x)):
    if i % 100 == 0:
        print(f"\r최대 길이 구하는 중 : {i/len(tokened_x) * 100:.2f}%, {i}/{len(tokened_x)}", end="")
    if max < len(tokened_x[i]):
        max = len(tokened_x[i])

print(f"\r최대 길이 구하는 중 : 100.00%")

print(f"가장 긴 문장의 길이 : {max}")

x_pad = pad_sequences(tokened_x, max)

X_train, X_test, Y_train, Y_test = train_test_split(
    x_pad, onehot_y, test_size=0.2
)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = np.array((X_train, X_test, Y_train, Y_test), dtype=object)
np.save("../learning_data/comment_data_max_{}_wordsize_{}.npy".format(max, wordsize), xy)