import pandas as pd
import numpy as np

from konlpy.tag import Okt  # 한국어 자연어 처리(형태소 분리기) 패키지

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle  # 파이썬 데이터형을 그대로 저장 하는 파일 저장 패키지

from tensorflow.keras.models import load_model

import os


comment_model_file = "../models/comment_category_classification_model_199_27766_0.8081.h5"  # 저장된 모델의 경로 지정
score_model_file = "../models/score_category_classification_model_199_14075_0.6085.h5"

comment_model = load_model(comment_model_file)
score_model = load_model(score_model_file)

max = int(os.path.splitext(comment_model_file)[0].split("/")[-1].split("_")[4])  # 파일 명을 이용 하여 최대 길이를 구함

validate_df = pd.read_csv("../validate_data/validate_data.csv")  # 확인용 데이터 셋 불러옴

print(validate_df.head())
print(validate_df.info())

X = validate_df["RawText"].copy()
comment_Y = validate_df["Polarity"]

# 모델에 입출력을 맞추기 위해 데이터 가공
# Y 전처리
with open("../models/label_encoder.pickle", "rb") as file:
    comment_label_encoder = pickle.load(file)

comment_label = comment_label_encoder.classes_

print(comment_label)

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
    comment_token = pickle.load(file)

comment_tokened_x = comment_token.texts_to_sequences(X)

# print(comment_tokened_x)

for i in range(len(comment_tokened_x)):
    if i % 100 == 0:
        print(f"\r최대 길이 구하는 중 : {i/len(comment_tokened_x) * 100:.2f}%", end="")
    if len(comment_tokened_x[i]) > max:
        comment_tokened_x[i] = comment_tokened_x[i][:max]

print("\r최대 길이 구하는 중 : 100.00%")

x_pad = pad_sequences(comment_tokened_x, max)

# 모델 정확도 확인
preds = comment_model.predict(x_pad)

predicts = []

for pred in preds:
    most = comment_label[np.argmax(pred)]
    predicts.append(most)

validate_df["Comment Pred"] = predicts

print(validate_df[["Polarity", "Comment Pred"]].head(20))

# print(df.head())

validate_df["Comment OX"] = 0

for i in range(len(validate_df)):
    if validate_df.loc[i, "Polarity"] == validate_df.loc[i, "Comment Pred"]:
        validate_df.loc[i, "Comment OX"] = "O"
    else:
        validate_df.loc[i, "Comment OX"] = "X"

print(validate_df["Comment OX"].value_counts())

print(validate_df["Comment OX"].value_counts() / len(validate_df))

# =================================================================

with open("../models/score_label_encoder.pickle", "rb") as file:
    score_label_encoder = pickle.load(file)

score_label = score_label_encoder.classes_

print(score_label)

with open("../models/score_word_token.pickle", "rb") as file:
    score_token = pickle.load(file)

score_tokened_x = score_token.texts_to_sequences(X)

for i in range(len(score_tokened_x)):
    if i % 100 == 0:
        print(f"\r최대 길이 구하는 중 : {i/len(score_tokened_x) * 100:.2f}%", end="")
    if len(score_tokened_x[i]) > max:
        score_tokened_x[i] = score_tokened_x[i][:max]

print("\r최대 길이 구하는 중 : 100.00%")

x_pad = pad_sequences(score_tokened_x, max)

# 모델 정확도 확인
preds = score_model.predict(x_pad)

predicts = []

for pred in preds:
    most = score_label[np.argmax(pred)]
    predicts.append(most)

validate_df["Score Pred"] = predicts

print(validate_df[["ReviewScore", "Score Pred"]].head(20))

validate_df["Score OX"] = 0

for i in range(len(validate_df)):
    if validate_df.loc[i, "ReviewScore"] == validate_df.loc[i, "Score Pred"]:
        validate_df.loc[i, "Score OX"] = "O"
    else:
        validate_df.loc[i, "Score OX"] = "X"

print(validate_df["Score OX"].value_counts())

print(validate_df["Score OX"].value_counts() / len(validate_df))

