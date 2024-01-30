import sys
from PyQt5.QtWidgets import *
from PyQt5 import uic

import pandas as pd
import numpy as np

from konlpy.tag import Okt  # 한국어 자연어 처리(형태소 분리기) 패키지

from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle  # 파이썬 데이터형을 그대로 저장 하는 파일 저장 패키지

from tensorflow.keras.models import load_model

import os


form_window = uic.loadUiType("../ui/comment_analysis.ui")[0]  # 디자인 파일 경로 입력


class Main(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 미리 학습된 모델의 파일 경로
        self.polarity_path = "../models/comment_category_classification_model_199_27766_0.7989.h5"
        self.score_path = "../models/score_category_classification_model_199_14075_0.6085.h5"
        self.stopwords = pd.read_csv("../stopwords.csv")

        # 모델을 불러옴
        self.polarity_model = load_model(self.polarity_path)
        self.score_model = load_model(self.score_path)

        # 파일 명을 이용 하여 필요한 정보를 처리
        self.max = int(os.path.splitext(self.polarity_path)[0].split("/")[-1].split("_")[4])

        self.okt = Okt()

        # 미리 저장해 놓은 encoder와 tokenizer를 불러드림
        with open("../models/label_encoder.pickle", "rb") as file:
            self.label_encoder = pickle.load(file)

        self.label = self.label_encoder.classes_

        print(self.label)

        with open("../models/score_label_encoder.pickle", "rb") as file:
            self.score_encoder = pickle.load(file)

        self.scores = self.score_encoder.classes_

        print(self.scores)

        with open("../models/word_token.pickle", "rb") as file:
            self.token = pickle.load(file)

        with open("../models/score_word_token.pickle", "rb") as file:
            self.score_token = pickle.load(file)

        # 버튼을 눌렀을 때 작동할 기능 연결
        self.btn_go.clicked.connect(self.comment_analysis_clicked_slot)
        self.btn_clear.clicked.connect(self.clear_clicked_slot)

    def comment_analysis_clicked_slot(self):
        X = self.tb_comment.toPlainText()  # text 박스에 있는 글을 읽어 옴

        print(X)
        if X != "":  # 공란이 아닌 경우에 분석 진행 
            X = self.okt.morphs(X, stem=True)

            words = []

            for j in range(len(X)):
                if len(X[j]) > 1:
                    if X[j] not in list(self.stopwords):
                        words.append(X[j])

            X = " ".join(words)

            tokened_x = self.token.texts_to_sequences([X])
            score_tokened_x = self.score_token.texts_to_sequences([X])

            if len(tokened_x[0]) > self.max:
                tokened_x = tokened_x[:self.max]

            if len(score_tokened_x[0]) > self.max:
                score_tokened_x = score_tokened_x[:self.max]

            x_pad = pad_sequences(tokened_x, self.max)
            score_x_pad = pad_sequences(score_tokened_x, self.max)

            preds = self.polarity_model.predict(x_pad)
            score_preds = self.score_model.predict(score_x_pad)

            self.tb_polarity.setText(str(self.label[np.argmax(preds)]))
            self.tb_score.setText(str(self.scores[np.argmax(score_preds)]))

        else:  # 공란인 경우에 정보 표시
            self.tb_comment.setPlainText("분석할 리뷰를 입력 하세요.")

    def clear_clicked_slot(self):
        self.tb_comment.setPlainText("")
        self.tb_polarity.setText("")
        self.tb_score.setText("")


if __name__ == "__main__":  # 파일 자체를 실행 시킨 경우
    app = QApplication(sys.argv)
    mainWindow = Main()
    mainWindow.show()
    sys.exit(app.exec_())
