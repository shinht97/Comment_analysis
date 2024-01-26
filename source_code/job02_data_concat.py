import pandas as pd
import glob

file_path = glob.glob("../datasets/*")

df = pd.DataFrame()

for i, file in enumerate(file_path):
    if i % 100 == 0:
        print(f"\rworking {i / len(file_path) * 100:.2f}%", end="")

    temp = pd.read_csv(file)  # csv를 읽어드림

    df = pd.concat([df, temp], axis="rows", ignore_index=True)  # dataframe을 합침

print(df.head())

df.to_csv("../learning_data/concated_file.csv", index=False)
