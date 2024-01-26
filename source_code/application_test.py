# 이 파일과 같은 동작 하는 GUI 프로그램 만들어 줘


import pandas as pd
import numpy as np

from konlpy.tag import Okt  # 한국어 자연어 처리(형태소 분리기) 패키지

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle  # 파이썬 데이터형을 그대로 저장 하는 파일 저장 패키지

from tensorflow.keras.models import load_model

import os


model_file = "../models/comment_category_classification_model_199_30131_0.8227196931838989.h5"

model = load_model(model_file)

stopwords = pd.read_csv("../stopwords.csv")

max = int(os.path.splitext(model_file)[0].split("/")[-1].split("_")[4])
wordsize = int(os.path.splitext(model_file)[0].split("/")[-1].split("_")[5])

with open("../models/label_encoder.pickle", "rb") as file:
    label_encoder = pickle.load(file)

label = label_encoder.classes_

print(label)

okt = Okt()

X = input("댓글 작성 : ")
X = okt.morphs(X, stem=True)

words = []
for j in range(len(X)):
    if len(X[j]) > 1:
        if X[j] not in list(stopwords):
            words.append(X[j])

X = " ".join(words)

with open("../models/word_token.pickle", "rb") as file:
    token = pickle.load(file)

tokened_x = token.texts_to_sequences([X])

# print(tokened_x)
# exit()

if len(tokened_x[0]) > max:
    tokened_x = tokened_x[:max]

x_pad = pad_sequences(tokened_x, max)

preds = model.predict(x_pad)

print(preds)

print(label[np.argmax(preds)])
