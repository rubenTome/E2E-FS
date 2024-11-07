import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

PATH = str(Path.cwd()) + "/"
fileNames = sorted(os.listdir(PATH))
fileNames = [file for file in fileNames if '.csv' in file]

usecols = ["duration", "emissions", "accuracy", "feature_mask"]
x = np.arange(len(usecols))
width = 0.15
multiplier = 0
keys = []

fig, ax = plt.subplots(layout='constrained')

acums = np.array([.0 for _ in range(len(usecols))])
for file in fileNames:
    df = pd.read_csv(PATH + file, usecols=usecols)
    print(file, df.values[0])
    acums += np.array(df.values[0])

for file in fileNames:
    df = pd.read_csv(PATH + file, usecols=usecols)
    keys = df.keys()

    offset = width * multiplier
    rects = ax.bar(x + offset, df.values[0] / acums, width, label=file)
    ax.bar_label(rects, padding=3)
    multiplier += 1

ax.set_ylabel('Values')
ax.set_title('Files')
ax.set_xticks(x + width, keys)
ax.legend(loc='upper left')
ax.set_ylim(0, 1)

plt.show()