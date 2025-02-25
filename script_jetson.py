from dataset_reader import colon, leukemia, lung181, lymphoma, dexter, gina, gisette, madelon
import numpy as np
import e2efs
import pandas as pd
from codecarbon import EmissionsTracker
import os
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import balanced_accuracy_score
import time

# script configuration
codecarbon_tracking = True
#numero de caracteristicas objetivo
n_features_to_select = 10
#epochs maximas sin reducir el numero de caracteristicas
wait = 25
#alfa inicial, step y final
initial_feature_importance = 0.005
feature_importance_step = 0.05
feature_importance = 0.505
k_folds = 3
#numero de repeticiones por experimento
N = 10
#implementacion del clasificador
networks = [None, "conv"]
datasets = [
    "leukemia",
    "lung",
    "lymphoma",
    "colon",
    "dexter", 
    #"gina", #accuracy demasiado baja
    "gisette", 
    #"madelon" #para este solo 5 caracteristicas
]
results_dir = "final_results_jetson"

precisions = ["16-true", "32", "64"]

if not os.path.exists(results_dir):
    os.mkdir(results_dir)

def decimal_range(start, stop, increment):
    while start < stop:
        yield start
        start += increment

if __name__ == '__main__':
    for precision in precisions:
        print("Precision:", precision)
        kfold = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=N, random_state=42)
        #select dataset
        for ds in datasets:
            if ds == "colon":
                selectedDs = colon
                print("selected colon dataset")
            elif ds == "leukemia":
                selectedDs = leukemia
                print("selected leukemia dataset")
            elif ds == "lung":
                selectedDs = lung181
                print("selected lung dataset")
            elif ds == "lymphoma":
                selectedDs = lymphoma
                print("selected lymphoma dataset")
            elif ds == "dexter":
                selectedDs = dexter
                print("selected dexter dataset")
            elif ds == "gina":
                selectedDs = gina
                print("selected gina dataset")
            elif ds == "gisette":
                selectedDs = gisette
                print("selected gisette dataset")
            elif ds == "madelon":
                selectedDs = madelon
                print("selected madelon dataset")
     
            for net in networks:
                if net == None:
                    netStr = ""
                else:
                    netStr = "_" + net
                if not os.path.exists(results_dir + "/results_" + ds + netStr):
                    os.mkdir(results_dir + "/results_" + ds + netStr)
                os.mkdir(results_dir + "/results_" + ds + netStr + "/fp" + precision)
                os.mkdir(results_dir + "/results_" + ds + netStr + "/fp" + precision + "/csv")
                os.mkdir(results_dir + "/results_" + ds + netStr + "/fp" + precision + "/stats")
     
                for fi in decimal_range(initial_feature_importance, feature_importance, feature_importance_step):
     
                    #set up directory names and csv columns
                    df = pd.DataFrame(columns=["test_acc", "balanced_acc", "nfeat", "max_alpha", "emissions", "duration"])
                    if net == "conv":
     
                        directory = results_dir + "/results_" + ds + "_conv/fp" + precision
                    else:
                        directory = results_dir + "/results_" + ds + "/fp" + precision
                    name = ds + "_a" + str(round(fi, 4)) + "_fp" + precision
                    csv_file = open(directory + "/csv/" + name + ".csv", "w")
                    f = open(directory + "/stats/" + name + ".txt", "w")
     
                    ## LOAD DATA
                    dataset = selectedDs.load_dataset()
                    raw_data = np.asarray(dataset['raw']['data'])
                    raw_label = np.asarray(dataset['raw']['label']).reshape(-1)
                    num_classes = len(np.unique(raw_label))
                    normalize = selectedDs.Normalize()
     
                    for j, (train_index, test_index) in enumerate(kfold.split(raw_data, raw_label)):
                        print('k_fold', j, 'of', k_folds*N)
     
                        train_data, train_label = raw_data[train_index], raw_label[train_index]
                        test_data, test_label = raw_data[test_index], raw_label[test_index]
     
                        train_data = normalize.fit_transform(train_data)
                        test_data = normalize.transform(test_data)
     
                        #if convolutional implementation is chosen
                        if net == "conv":
                            print("SELECTED CONV IMPLEMENTATION")
                        else:
                            print("SELECTED LINEAR IMPLEMENTATION")
     
                        train_label = np.array(train_label).astype(int)
                        test_label = np.array(test_label).astype(int)
     
                        print("features to select:", n_features_to_select)
     
                        #kfold rep tracker starts monitoring here
                        if codecarbon_tracking:
                            tracker = EmissionsTracker(measure_power_secs=1, log_level="critical", tracking_mode="process", gpu_ids=[0])
                            tracker.start()
                        else:
                            tracker = time.time()
     
                        ## LOAD E2EFSSoft model
                        model = e2efs.E2EFSSoft(n_features_to_select=n_features_to_select, wait=wait, feature_importance=fi, precision=precision, network=net)
                        ## FIT THE SELECTION
                        model.fit(train_data, train_label, validation_data=(test_data, test_label), batch_size=2, max_epochs=2000)
                        ## FINETUNE THE MODEL
                        #model.fine_tune(train_data, train_label, validation_data=(test_data, test_label), batch_size=2, max_epochs=100)
     
                        #kfold rep tracker stops monitoring here
                        if codecarbon_tracking:
                            tracker.stop()
                            csvf = pd.read_csv("emissions.csv")
                            emissions = csvf["emissions"].values[0]
                            duration = csvf["duration"].values[0]
                            os.remove("emissions.csv")
                        else:
                            emissions = -1
                            duration = time.time() - tracker
     
                        ## GET THE MODEL RESULTS
                        metrics = model.evaluate(test_data, test_label)
                        print(metrics)
                        predicted = model.predict(test_data)
                        predicted = [np.argmax(i) for i in predicted]
                        balanced_acc = balanced_accuracy_score(predicted, test_label)
                        print("BALANCED ACCURACY:", balanced_acc)
                        ## GET THE MASK
                        mask = model.get_mask()
                        print('MASK:', mask)
                        ## GET THE RANKING
                        ranking = model.get_ranking()
                        print('RANKING:', ranking)
                        nf = model.get_nfeats()
                        print("NUMBER OF FEATURES:", nf)
                        print("ALPHA MAX:", fi)
                        if j > 0:
                            df.loc[j] = [round(metrics["test_accuracy"], 4), round(balanced_acc, 4), nf, fi, emissions, duration]
                            df.to_csv(directory + "/csv/" + name + ".csv", index=False)
     
                    #write stats and global emissions
                    f.write(df.describe().to_string())