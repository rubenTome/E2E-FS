import pandas as pd
from pathlib import Path
import os

PATH = str(Path.cwd()) + "/"

script = "example.py"
output = "emissions_float32.csv"
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
    df = pd.read_csv(PATH + output, usecols=usecols)
    acum["duration"].append(df["duration"])
    acum["emissions"].append(df["emissions"])
    acum["accuracy"].append(df["accuracy"])
    acum["feature_mask"].append(df["feature_mask"])

df = pd.DataFrame(columns=usecols)

for i in range(len(usecols)):
    df[usecols[i]] = sum(acum[usecols[i]]) / len(acum[usecols[i]])

df.to_csv(script.split(".")[0] + "_mean" + str(n) + ".csv", index=False)

if deletebaks:
    for item in os.listdir(PATH):
        if item.endswith(".bak"):
            os.remove(os.path.join(PATH, item))