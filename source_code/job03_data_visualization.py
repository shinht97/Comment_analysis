import pandas as pd
import matplotlib.pyplot as plt

polarity_df = pd.read_csv("../learning_data/concated_file.csv")

plt.figure(1)
bar = plt.bar(["Pos", "soso", "Neg"], polarity_df["Polarity"].value_counts().values)

for rect in bar:
    height = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height, height, ha="center", va="center", size=10)

x = [i * 10 for i in range(0, 11)]

plt.figure(2)
bar2 = plt.bar(x, polarity_df["ReviewScore"].value_counts(ascending=True).values)

for rect in bar2:
    height2 = rect.get_height()
    plt.text(rect.get_x() + rect.get_width() / 2, height2, height2, ha="center", va="center", size=10)

plt.show()
