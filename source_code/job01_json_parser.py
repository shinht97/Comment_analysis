import json
import glob
import pandas as pd
import os

file_path = glob.glob("../raw_data/*/*")  # 특정 폴더 하위에 있는 모든 파일을 찾음

for k in range(0, len(file_path)):  # 각 파일 경로 당 작업 수행

    if k % 100 == 0:
        print(f"\r파일 읽어 오는 중 : {k / len(file_path) * 100:.2f}%", end="")  # 작업 진행 정도 표시

    RawText = []
    Polarity = []  # 빈 리스트 생성

    with open(file_path[k], "r", encoding="utf-8") as file:  # 파일을 염
        json_data = json.load(file)  # json 형태의 파일을 읽어서

        for i in range(len(json_data)):
            RawText.append(json_data[i]["RawText"])  # 리뷰 자체를 리스트에 추가

            polarity = 0

            for j in range(len(json_data[i]["Aspects"])):
                polarity += int(json_data[i]["Aspects"][j]["SentimentPolarity"])  # 긍정 부정 반응을 요소를 계산

            if polarity >= len(json_data[i]["Aspects"]) * 0.7:  # 전체 반응 개수 중에 70%로 긍정이면
                Polarity.append("긍정")  # 긍정
            elif (polarity < len(json_data[i]["Aspects"]) * 0.7) and (polarity >= 0):  # 그 이하면 무반은
                Polarity.append("무반응")  # soso
            elif polarity < 0:  # 음수면 부정
                Polarity.append("부정")  # 부정

    file_name = os.path.splitext(file_path[k])[0].split("\\")[-1]  # 파일 경로에서 파일 명을 분리

    df = pd.DataFrame()
    df["RawText"] = RawText
    df["Polarity"] = Polarity

    df.to_csv(f"../datasets/{file_name}.csv", index=False)  # dataframe 저장

print(f"\r파일 읽어 오는 중 : 100.00%")