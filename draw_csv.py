import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

PATH = str(Path.cwd()) + "/"
fileNames = os.listdir(PATH)
fileNames = [file for file in fileNames if '.csv' in file]

usecols = [0, 1, 5, 6]
x = np.arange(len(usecols))
width = 0.25
multiplier = 0
keys = []

fig, ax = plt.subplots(layout='constrained')

for file in fileNames:
    fileId = "_" + file.split("_")[1].split(".")[0]
    df = pd.read_csv(PATH + file, usecols=usecols)
    keys = df.keys()
    df["accuracy"] = df["accuracy"] * 100
    df["duration"] = df["duration"] / 10
    df["emissions"] = df["emissions"] * 10000
    # plt.bar(df.keys() + fileId, df.values[0])

    offset = width * multiplier
    rects = ax.bar(x + offset, df.values[0], width, label=file)
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel('Values')
ax.set_title('Files')
ax.set_xticks(x + width, keys)
ax.legend(loc='upper left', ncols=3)
ax.set_ylim(0, 250)

plt.show()