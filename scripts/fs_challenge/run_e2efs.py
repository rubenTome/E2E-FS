path = "./scripts/fs_challenge"

datasets = [
    'gina',
    'dexter',
    'gisette',
    'madelon'
]

for dataset in datasets:

    print("EXECUTING", dataset)

    if dataset == "madelon":
        exec(open(path + "/" + dataset + "/script_e2efs_nn.py").read())
    else:
        exec(open(path + "/" + dataset + "/script_e2efs.py").read())