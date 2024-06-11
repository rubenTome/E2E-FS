path = "./scripts/fs_challenge"

datasets = [
    #'gina',
    'dexter',
    'gisette',
    'madelon'
]

scripts = [
    "script_e2efs.py",
    "script_e2efs_nn.py",
    "script_e2efs_ranking.py",
    "script_e2efs_ranking_nn.py"
]

for dataset in datasets:
    for script in scripts:
        print("EXECUTING", dataset)
        if dataset == "dexter" and script == "script_e2efs_ranking_nn.py":
            continue
        if dataset == "gina" and (script == "script_e2efs_nn.py" or script == "script_e2efs_ranking_nn.py"):
            continue
        if dataset == "gisette" and (script == "script_e2efs_nn.py" or script == "script_e2efs_ranking_nn.py"):
            continue        
        if dataset == "madelon" and (script == "script_e2efs.py" or script == "script_e2efs_ranking"):
            continue
        exec(open(path + "/" + dataset + "/" + script).read())