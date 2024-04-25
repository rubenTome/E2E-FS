from codecarbon import EmissionsTracker
from backend_config import outputFileName
tracker = EmissionsTracker(log_level="warning", output_file=outputFileName)
tracker.start()
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras import optimizers
from e2efs import models
import numpy as np
from keras import backend as K
from backend_config import bcknd, ops, n_features_to_select, nclasses, loss, batch_size, model_fun, dataset
import pandas as pd
from src.wrn.network_models import three_layer_nn_q, wrn164, three_layer_nn
from dataset_reader import colon, leukemia, lung181, lymphoma
from keras.datasets import mnist


K.backend = bcknd
if model_fun == "three_layer_nn":
    model_fun = three_layer_nn
elif model_fun == "wrn164":
    model_fun = wrn164
else:
    raise Exception("invalid model function")
if dataset == "mnist":
    dataset = mnist.load_data
elif dataset == "colon":
    dataset = colon.load_dataset
elif dataset == "leukemia":
    dataset = leukemia.load_dataset
elif dataset == "lung181":
    dataset = lung181.load_dataset
elif dataset == "lymphoma":
    dataset = lymphoma.load_dataset
else:
    raise Exception("Invalid dataset")
#colon, lung en float16 poca precision, no baja nnz

if __name__ == '__main__':

    ## LOAD DATA
    if dataset == mnist.load_data:
        (x_train, y_train), (x_test, y_test) = dataset()
        x_train = np.expand_dims(x_train, axis=-1)
        x_test = np.expand_dims(x_test, axis=-1)
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
    else:
        ds = dataset()
        data = ds["raw"]["data"]
        label = ds['raw']["label"]
        #divide a la mitad el dataset para crear train y test
        x_train = data[:int(len(data) / 2)]
        x_test = data[int(len(data) / 2):]
        mean_data = x_train.mean(axis=0)
        std_data = x_train.std(axis=0) + 1e-8
        x_train = (x_train - mean_data)/std_data
        x_test = (x_test - mean_data)/std_data
        y_train = to_categorical(label[:int(len(label) / 2)], num_classes=2)
        y_test = to_categorical(label[int(len(label) / 2):], num_classes=2)

    ## LOAD MODEL AND COMPILE IT (NEVER FORGET TO COMPILE!)
    # model = three_layer_nn_q(input_shape=x_train.shape[1:], nclasses=10, regularization=5e-4, layer_dims=[50, 25, 10])
    model = model_fun(input_shape=x_train.shape[1:], nclasses=nclasses, regularization=5e-4)
    model.compile(optimizer=optimizers.SGD(), metrics=['acc'], loss=loss, run_eagerly=True)
    # model.fit(
    #    x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2
    # )

    ## LOAD E2EFS AND RUN IT
    fs_class = models.E2EFSSoft(n_features_to_select=n_features_to_select).attach(model).fit(
        x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2
    )
    ## OPTIONAL: LOAD E2EFS AND RUN IT
    # fs_class = models.E2EFS(n_features_to_select=39).attach(model).fit(
    #     x_train, y_train, batch_size=128, validation_data=(x_test, y_test), verbose=2
    # )

    ## OPTIONAL: LOAD E2EFSRanking AND RUN IT (do not use fine tuning with this model, only get_ranking)
    # fs_class = models.E2EFSRanking().attach(model).fit(
    #     x_train, y_train, batch_size=128, validation_data=(x_test, y_test), verbose=2
    # )

    ## FINE TUNING
    def scheduler(epoch):
        if epoch < 20:
            return .1
        elif epoch < 40:
            return .02
        elif epoch < 50:
            return .004
        else:
            return .0008

    fs_class.fine_tuning(x_train, y_train, epochs=60, batch_size=128, validation_data=(x_test, y_test),
                         callbacks=[LearningRateScheduler(scheduler)], verbose=2)
    print('FEATURE_RANKING :', fs_class.get_ranking())
    acc = fs_class.get_model().evaluate(x_test, y_test, batch_size=128)[-1]
    print('ACCURACY : ', acc)
    nnz = np.count_nonzero(fs_class.get_mask())
    print('FEATURE_MASK NNZ :', nnz)

tracker.stop()

df = pd.read_csv(outputFileName)
df["accuracy"] = acc
df["feature_mask"] = nnz
df.to_csv(outputFileName, index=False)
