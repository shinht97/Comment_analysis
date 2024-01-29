import pandas as pd
import numpy as np

from konlpy.tag import Okt  # 한국어 자연어 처리(형태소 분리기) 패키지

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle  # 파이썬 데이터형을 그대로 저장 하는 파일 저장 패키지

from tensorflow.keras.models import load_model

import os


model_file = "../models/comment_category_classification_model_199_30131_0.8285316824913025.h5"  # 저장된 모델의 경로 지정

model = load_model(model_file)

max = int(os.path.splitext(model_file)[0].split("/")[-1].split("_")[4])  # 파일명을 이용하여 최대 길이를 구함
wordsize = int(os.path.splitext(model_file)[0].split("/")[-1].split("_")[5])  # 파일명에서 단어의 개수를 구함

df = pd.read_csv("../validate_data/validate_data.csv")  # 확인용 데이터 셋 불러옴

print(df.head())
print(df.info())

X = df["RawText"]
Y = df["Polarity"]

# 모델에 입출력을 맞추기 위해 데이터 가공
# Y 전처리
with open("../models/label_encoder.pickle", "rb") as file:
    label_encoder = pickle.load(file)

label = label_encoder.classes_

print(label)

# X 전처리
okt = Okt()

for i in range(len(X)):
    if i % 100 == 0:
        print(f"\r형태소 처리 중 : {i / len(X) * 100:.2f}%", end="")

    X[i] = okt.morphs(X[i], stem=True)

stopwords = pd.read_csv("../stopwords.csv")

print("\r형태소 처리 중 : 100.00%")

for i in range(len(X)):
    if i % 100 == 0:
        print(f"\r문자열 필터링 중 : {i/len(X) * 100:.2f}%", end="")
    words = []
    for j in range(len(X[i])):
        if len(X[i][j]) > 1:
            if X[i][j] not in list(stopwords):
                words.append(X[i][j])

    X[i] = " ".join(words)

print("\r문자열 필터링 중 : 100.00%")

with open("../models/word_token.pickle", "rb") as file:
    token = pickle.load(file)

tokened_x = token.texts_to_sequences(X)

print(tokened_x)

for i in range(len(tokened_x)):
    if i % 100 == 0:
        print(f"\r최대 길이 구하는 중 : {i/len(tokened_x) * 100:.2f}%", end="")
    if len(tokened_x[i]) > max:
        tokened_x[i] = tokened_x[i][:max]

print("\r최대 길이 구하는 중 : 100.00%")

x_pad = pad_sequences(tokened_x, max)

# 모델 정확도 확인
preds = model.predict(x_pad)

predicts = []

for pred in preds:
    most = label[np.argmax(pred)]
    predicts.append(most)

df["Predict"] = predicts

print(df.head())

df["OX"] = 0

for i in range(len(df)):
    if df.loc[i, "Polarity"] == df.loc[i, "Predict"]:
        df.loc[i, "OX"] = "O"
    else:
        df.loc[i, "OX"] = "X"

print(df["OX"].value_counts())

print(df["OX"].value_counts() / len(df))
