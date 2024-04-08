import csv
import pandas as pd
from pathlib import Path

PATH = str(Path.cwd()) + "/"

script = "example.py"
output = "emissions_float32.csv"
usecols = ["duration", "emissions", "accuracy", "feature_mask"]
n = 10
acum = {
    "duration":[],
    "emissions":[],
    "accuracy":[],
    "feature_mask":[]
}

for _ in range(n):
    exec(open(script).read())
    df = pd.read_csv(PATH + output, usecols=usecols)
    acum["duration"].append(df["duration"])
    acum["emissions"].append(df["emissions"])
    acum["accuracy"].append(df["accuracy"])
    acum["feature_mask"].append(df["feature_mask"])

for i in range(len(usecols)):
    df = pd.DataFrame(columns=usecols)
    df[usecols[i]] = sum(acum[usecols[i]]) / len(acum[usecols[i]])

df.to_csv(script.split(".")[1] + "_mean" + str(n) + ".csv", index=False)