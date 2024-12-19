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

n_features_to_select = 10
feature_importance = 0.6
wait = 1
k_folds = 3
N = 10
precision = sys.argv[1]
print("Precision:", precision)
set_precision(precision)
kfold = RepeatedStratifiedKFold(n_splits=k_folds, n_repeats=N, random_state=42)
networks = [None, "conv"]
datasets = [
    "colon",
    "leukemia", 
    "lung", 
    "lymphoma", 

    "dexter", 
    # "gina", 
    # "gisette", 
    # "madelon"
]

def decimal_range(start, stop, increment):
    while start < stop:
        yield start
        start += increment

if __name__ == '__main__':
    for ds in datasets:
        if ds == "colon":
            selectedDs = colon
        elif ds == "leukemia":
            selectedDs = leukemia
        elif ds == "lung":
            selectedDs = lung181
        elif ds == "lymphoma":
            selectedDs = lymphoma
        elif ds == "dexter":
            selectedDs = dexter
        elif ds == "gina":
            selectedDs = gina
        elif ds == "gisette":
            selectedDs = gisette
        elif ds == "madelon":
            selectedDs = madelon
        for net in networks:
            for fi in decimal_range(.0, feature_importance, 0.1):
                
                df = pd.DataFrame(columns=["test_acc", "balanced_acc", "nfeat", "max_alpha", "emissions", "duration"])
                name = ds + "_a" + str(round(fi, 4)) + "_fp" + precision
                if net == "conv":
                    directory = "results_" + ds + "_conv/fp" + precision
                else:
                    directory = "results_" + ds + "/fp" + precision
                f = open(directory + "/stats/" + name + ".txt", "w")

                ## LOAD DATA
                dataset = selectedDs.load_dataset()
                raw_data = np.asarray(dataset['raw']['data'])
                raw_label = np.asarray(dataset['raw']['label']).reshape(-1)
                num_classes = len(np.unique(raw_label))
                normalize = selectedDs.Normalize()
                globalTracker = EmissionsTracker(tracking_mode="process", output_file="gemissions.csv")
                globalTracker.start()
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

                    valid_features = np.where(np.abs(train_data).sum(axis=0) > 0)[0]
                    if len(valid_features) < train_data.shape[1]:
                        print('Removing', train_data.shape[1] - len(valid_features), 'zero features')
                        train_data = train_data[:, valid_features]
                        test_data = test_data[:, valid_features]

                    tracker = EmissionsTracker(tracking_mode="process")
                    tracker.start()
                    ## LOAD E2EFSSoft model
                    model = e2efs.E2EFSSoft(n_features_to_select=n_features_to_select, feature_importance=fi, network=net)
                    ## FIT THE SELECTION
                    model.fit(train_data, train_label, validation_data=(test_data, test_label), batch_size=2, max_epochs=2000, wait=wait)
                    ## FINETUNE THE MODEL
                    #model.fine_tune(train_data, train_label, validation_data=(test_data, test_label), batch_size=2, max_epochs=100)
                    tracker.stop()
                    csvf = pd.read_csv("emissions.csv")
                    emissions = csvf["emissions"].values[0]
                    duration = csvf["duration"].values[0]
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
                    #discard first k_fold rep
                    if j > 0:
                        df.loc[j] = [round(metrics["test_accuracy"], 4), round(balanced_acc, 4), nf, fi, emissions, duration]
                    os.remove("emissions.csv")
                    df.to_csv(directory + "/csv/" + name + ".csv", index=False)
                f.write(df.describe().to_string())
                globalTracker.stop()
                gcsvf = pd.read_csv("gemissions.csv")
                gemissions = csvf["emissions"].values[0]
                f.write("\nGLOBAL EMISSIONS: " + str(emissions) + " ( " + str(gemissions / (k_folds * N)) + " each execution)")
                os.remove("gemissions.csv")