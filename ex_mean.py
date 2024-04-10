import pandas as pd
from pathlib import Path
import os
from backend_config import script, outputFileName

PATH = str(Path.cwd()) + "/"

usecols = ["duration", "emissions", "accuracy", "feature_mask"]
acum = {
    "duration":[],
    "emissions":[],
    "accuracy":[],
    "feature_mask":[]
}
deletebaks = True
n = 10

print("EXECUTING ", script)

for i in range(n):
    print("LOOP: N=", i)
    exec(open(script).read())
    df = pd.read_csv(PATH + outputFileName, usecols=usecols)
    acum["duration"].append(df["duration"])
    acum["emissions"].append(df["emissions"])
    acum["accuracy"].append(df["accuracy"])
    acum["feature_mask"].append(df["feature_mask"])

df = pd.DataFrame(columns=usecols)

for i in range(len(usecols)):
    df[usecols[i]] = sum(acum[usecols[i]]) / len(acum[usecols[i]])

df.to_csv(outputFileName.split(".")[0] + "_mean" + str(n) + ".csv", index=False)

if deletebaks:
    for item in os.listdir(PATH):
        if item.endswith(".bak"):
            os.remove(os.path.join(PATH, item))