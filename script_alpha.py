from dataset_reader import colon, leukemia, lung181, lymphoma, dexter, gina, gisette, madelon
import numpy as np
import e2efs
import pandas as pd
from codecarbon import EmissionsTracker
import os
import sys
from src.precision import set_precision
from sklearn.model_selection import RepeatedStratifiedKFold
from keras.utils import to_categorical
from sklearn.metrics import balanced_accuracy_score
from torch import nn
import time

# script configuration
codecarbon_tracking = True
factor = None #.5
fixed_nfeat = 100
feature_importance = 0.6
wait = 1
k_folds = 3
N = 15
precision = sys.argv[1]
print("Precision:", precision)
set_precision(precision)
kfold = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=N, random_state=42)
networks = [None]#, "conv"]
datasets = [
    "leukemia",
    "lung",
    "lymphoma",
    "colon",
    # "dexter", 
    # "gina", 
    "gisette", 
    "madelon"
]
results_dir = "final_results"
if not os.path.exists(results_dir):
    os.mkdir(results_dir)

def decimal_range(start, stop, increment):
    while start < stop:
        yield start
        start += increment

if __name__ == '__main__':

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

        if not os.path.exists(results_dir + "/results_" + ds):
            os.mkdir(results_dir + "/results_" + ds)
        os.mkdir(results_dir + "/results_" + ds + "/fp" + precision)
        os.mkdir(results_dir + "/results_" + ds + "/fp" + precision + "/csv")
        os.mkdir(results_dir + "/results_" + ds + "/fp" + precision + "/stats")

        for net in networks:
            for fi in decimal_range(.0, feature_importance, 0.1):
                
                #set up directory names and csv columns
                df = pd.DataFrame(columns=["test_acc", "balanced_acc", "nfeat", "max_alpha", "emissions", "duration"])
                if net == "conv":
                    
                    directory = results_dir + "/results_" + ds + "_conv/fp" + precision
                else:
                    directory = results_dir + "/results_" + ds + "/fp" + precision
                name = ds + "_a" + str(round(fi, 4)) + "_f" + str(factor) + "_fp" + precision
                csv_file = open(directory + "/csv/" + name + ".csv", "w")
                f = open(directory + "/stats/" + name + ".txt", "w")

                ## LOAD DATA
                dataset = selectedDs.load_dataset()
                raw_data = np.asarray(dataset['raw']['data'])
                raw_label = np.asarray(dataset['raw']['label']).reshape(-1)
                num_classes = len(np.unique(raw_label))
                normalize = selectedDs.Normalize()

                #global tracker starts monitoring here
                if codecarbon_tracking:
                    globalTracker = EmissionsTracker(output_file="gemissions.csv", measure_power_secs=1, log_level="critical")
                    globalTracker.start()
                else:
                    globalTracker = time.time()
                
                for j, (train_index, test_index) in enumerate(kfold.split(raw_data, raw_label)):
                    print('k_fold', j, 'of', k_folds*N)

                    train_data, train_label = raw_data[train_index], raw_label[train_index]
                    test_data, test_label = raw_data[test_index], raw_label[test_index]

                    train_data = normalize.fit_transform(train_data)
                    test_data = normalize.transform(test_data)

                    #if convolutional implementation is chosen
                    if net == "conv":
                        print("SELECTED CONV IMPLEMENTATION")
                        train_data = train_data[:, :, np.newaxis]
                        test_data = test_data[:, :, np.newaxis]
                        train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
                        test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
                    else:
                        print("SELECTED LINEAR IMPLEMENTATION")      

                    valid_features = np.where(np.abs(train_data).sum(axis=0) > 0)[0]
                    if len(valid_features) < train_data.shape[1]:
                        print('Removing', train_data.shape[1] - len(valid_features), 'zero features')
                        train_data = train_data[:, valid_features]
                        test_data = test_data[:, valid_features]

                    if factor != None:
                        n_features_to_select = int(factor * int(np.prod(train_data.shape[1:])))
                    else:
                        n_features_to_select = fixed_nfeat
                    print("features to select:", n_features_to_select)
                    
                    #kfold rep tracker starts monitoring here
                    if codecarbon_tracking:
                        tracker = EmissionsTracker(measure_power_secs=1, log_level="critical", tracking_mode="process")
                        tracker.start()
                    else:
                        tracker = time.time()
                    
                    ## LOAD E2EFSSoft model
                    model = e2efs.E2EFSSoft(n_features_to_select=n_features_to_select, feature_importance=fi, network=net)
                    ## FIT THE SELECTION
                    model.fit(train_data, train_label, validation_data=(test_data, test_label), batch_size=2, max_epochs=2000, wait=wait)
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
                if codecarbon_tracking:
                    globalTracker.stop()
                    gcsvf = pd.read_csv("gemissions.csv")
                    gemissions = csvf["emissions"].values[0]
                    f.write("\nGLOBAL EMISSIONS: " + str(emissions) + " ( " + str(gemissions / (k_folds * N)) + " each execution)")
                    os.remove("gemissions.csv")
                else:
                    globalTracker = time.time() - globalTracker
                    f.write("\nGLOBAL EMISSIONS: " + str(globalTracker) + " ( " + str(globalTracker / (k_folds * N)) + " each execution)")

