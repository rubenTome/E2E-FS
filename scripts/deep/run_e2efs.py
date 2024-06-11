import os

path = os.path.dirname(os.path.realpath(__file__))

datasets = [
    "cifar10", 
    "cifar100", 
    "fashion_mnist", 
    "mnist"
]
scripts = [
    "script_e2efs",
    "script_e2efs_ranking",
    "script_e2efs_transfer",
    "script_e2efs_ranking_transfer"
]

for dataset in datasets:
    for script in scripts:
        if dataset == "cifar10" and (script == "script_e2efs_transfer" or script == "script_e2efs_ranking_transfer"):
            continue
        if dataset == "cifar100" and script == "script_e2efs_transfer":
            continue
        exec(open(path + "/" + dataset + "/" + script).read())