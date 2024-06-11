import os

path = os.path.dirname(os.path.realpath(__file__))

datasets = [
    "colon", 
    "leukemia", 
    "lung181", 
    "lymphoma"
]
scripts = [
    "script_e2efs",
    "script_e2efs_ranking"
]

for dataset in datasets:
    for script in scripts:
        exec(open(path + "/" + dataset + "/" + script).read())