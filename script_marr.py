from dataset_reader import colon
import numpy as np
import e2efs
import pandas as pd
from codecarbon import EmissionsTracker
import os
import sys
from src.precision import set_precision

n_features_to_select = 10
feature_importance = 0.5
wait = 1
N = 20
precision = sys.argv[1]
print("Precision:", precision)
set_precision(precision)

def decimal_range(start, stop, increment):
    while start < stop:
        yield start
        start += increment

for fi in decimal_range(.0, feature_importance, 0.05):
    df = pd.DataFrame(columns=["test_acc", "nfeat", "max_alpha", "emissions", "duration"])
    name = "colon_a" + str(round(fi, 4)) + "_fp" + precision
    directory = "results/fp" + precision
    f = open(directory + "/colon/stats/" + name + ".txt", "w")
    for n in range(N):
        tracker = EmissionsTracker(tracking_mode="process")
        tracker.start()
        if __name__ == '__main__':
            ## LOAD DATA
            dataset = colon.load_dataset()
            raw_data = np.asarray(dataset['raw']['data'])
            raw_label = np.asarray(dataset['raw']['label']).reshape(-1)
            train_data = raw_data[:int(len(raw_data) * 0.8)]
            train_label = raw_label[:int(len(raw_label) * 0.8)]
            test_data = raw_data[int(len(raw_data) * 0.8):]
            test_label = raw_label[int(len(raw_label) * 0.8):]
            normalize = colon.Normalize()
            train_data = normalize.fit_transform(train_data)
            test_data = normalize.transform(test_data)
            ## LOAD E2EFSSoft model
            model = e2efs.E2EFSSoft(n_features_to_select=n_features_to_select, feature_importance=fi)
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
            ## GET THE MASK
            mask = model.get_mask()
            print('MASK:', mask)
            ## GET THE RANKING
            ranking = model.get_ranking()
            print('RANKING:', ranking)
            nf = model.get_nfeats()
            print("NUMBER OF FEATURES:", nf)
            print("ALPHA MAX:", fi)
            df.loc[n] = [round(metrics["test_accuracy"], 4), nf, fi, emissions, duration]
            os.remove("emissions.csv") 
            df.to_csv(directory + "/colon/csv/" + name + ".csv", index=False)
    f.write(df.describe().to_string())