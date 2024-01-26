import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from konlpy.tag import Okt

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

import pickle


df = pd.read_csv("../learning_data/concated_file.csv")  # 파일을 읽어와 dataframe으로 만듦

X = df["RawText"]  # 학습 데이터 : 댓글 
Y = df["Polarity"]  # 결과 : 반은

# Y 전처리
label_encoder = LabelEncoder()
labeled_y = label_encoder.fit_transform(Y)  # 라벨을 이용하여 one hot encoding 진행

label = label_encoder.classes_  # 만들어진 라벨 확인

print(label)

with open("../models/label_encoder.pickle", "wb") as file:
    pickle.dump(label_encoder, file)  # 만들어진 라벨을 저장

onehot_y = to_categorical(labeled_y)

# X 전처리
okt = Okt()

for i in range(len(X)):  # 모든 X에 대해
    if i % 100 == 0:
        print(f"\r형태소 분리 중 : {i/len(X) * 100:.2f}%, {i}/{len(X)}", end="")  # 진행 상황 표시
    
    X[i] = okt.morphs(X[i], stem=True)  # 각 문장을 형태소로 분리
    
print(f"\r형태소 분리 중 : 100.00%")

stopwords = pd.read_csv("../stopwords.csv", index_col=0)  # 한 글자나 감탄사 등 학습을 저해 하는 단어들의 리스트

for i in range(len(X)):  # 모든 X에 대해
    if i % 100 == 0:
        print(f"\r문자열 필터링 중 : {i/len(X) * 100:.2f}%, {i}/{len(X)}", end="")
        
    words = []
    
    for j in range(len(X[i])):
        if len(X[i][j]) > 1:  # 만일 길이가 1보다 크고
            if X[i][j] not in list(stopwords):  # 학습을 저해하는 단어가 아닌 경우
                words.append(X[i][j])

    X[i] = " ".join(words)  # 하나의 문장으로 만들어줌

print(f"\r문자열 필터링 중 : 100.00%")

token = Tokenizer()  # 토큰화 객체 생성
token.fit_on_texts(X)
tokened_x = token.texts_to_sequences(X)

wordsize = len(token.word_index) + 1  # 기존에 없거나, 문제가 발생 하는 토큰을 위한 0을 추가

print(f"word size : {wordsize}")

with open("../models/word_token.pickle", "wb") as file:
    pickle.dump(token, file)

max = 0

for i in range(len(tokened_x)):
    if i % 100 == 0:
        print(f"\r최대 길이 구하는 중 : {i/len(tokened_x) * 100:.2f}%, {i}/{len(tokened_x)}", end="")
        
    if max < len(tokened_x[i]):  # 모든 X에 대해 가장 긴문장을 찾음
        max = len(tokened_x[i])

print(f"\r최대 길이 구하는 중 : 100.00%")

print(f"가장 긴 문장의 길이 : {max}")

x_pad = pad_sequences(tokened_x, max)  # 가장 긴 문장에 맞춰 모든 데이터의 길이를 통일

X_train, X_test, Y_train, Y_test = train_test_split(
    x_pad, onehot_y, test_size=0.2
)  # 학습과 테스트용으로 데이터를 분리

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

xy = np.array((X_train, X_test, Y_train, Y_test), dtype=object)
np.save("../learning_data/comment_data_max_{}_wordsize_{}.npy".format(max, wordsize), xy)  # 데이터 저장
