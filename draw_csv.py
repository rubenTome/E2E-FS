import os
import pandas as pd
import matplotlib.pyplot as plt

PATH = '/home/lidia/Documents/ruben/E2E-FS/' # Use your path
fileNames = os.listdir(PATH)
fileNames = [file for file in fileNames if '.csv' in file]

for file in fileNames:
    df = pd.read_csv(PATH + file, index_col = 0, usecols=[0, 1, 5, 6])
    plt.plot(df)

plt.show()