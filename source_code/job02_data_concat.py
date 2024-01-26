import pandas as pd
import glob

file_path = glob.glob("../datasets/*")  # 데이터가 있는 경로를 지정하여, 하위 파일을 리스트로 반환

df = pd.DataFrame()

for i, file in enumerate(file_path):  # 리스트에 있는 모든 파일에 대해
    if i % 100 == 0:
        print(f"\rworking {i / len(file_path) * 100:.2f}%", end="")  # 진행도 표시

    temp = pd.read_csv(file)  # csv를 읽어드림

    df = pd.concat([df, temp], axis="rows", ignore_index=True)  # dataframe을 합침

print(f"\rworking 100.00%")

print(df.head())

df.to_csv("../learning_data/concated_file.csv", index=False)  # 합친 dataframe을 저장
