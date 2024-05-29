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
from backend_config import bcknd, ops, selected_dataset, loss, epochs
import pandas as pd
from src.wrn.network_models import three_layer_nn_v2, wrn164, three_layer_nn
from dataset_reader import colon, leukemia, lung181, lymphoma
from keras.datasets import mnist, cifar10, fashion_mnist
import keras

ops.cast_to_floatx = lambda x: ops.cast(x, keras.config.floatx())
K.backend = bcknd

print("model function:", selected_dataset["model"])
if selected_dataset["model"] == "three_layer_nn":
    model_fun = three_layer_nn
elif selected_dataset["model"] == "wrn164":
    model_fun = wrn164
elif selected_dataset["model"] == "three_layer_nn_v2":
    model_fun = three_layer_nn_v2
else:
    raise Exception("invalid model function")

microarr = False
print("used dataset:", selected_dataset["name"])
if selected_dataset["name"] == "mnist":
    dataset = mnist.load_data
elif selected_dataset["name"] == "cifar10":
    dataset = cifar10.load_data
elif selected_dataset["name"] == "fashion_mnist":
    dataset = fashion_mnist.load_data
elif selected_dataset["name"] == "colon":
    microarr = True
    dataset = colon.load_dataset
elif selected_dataset["name"] == "leukemia":
    microarr = True
    dataset = leukemia.load_dataset
elif selected_dataset["name"] == "lung181":
    microarr = True
    dataset = lung181.load_dataset
elif selected_dataset["name"] == "lymphoma":
    microarr = True
    dataset = lymphoma.load_dataset
else:
    raise Exception("Invalid dataset", selected_dataset["name"])

if __name__ == '__main__':

    ## LOAD DATA
    # if temporal, para diferenciar microarray de los demas conjuntos de datos
    if not microarr:
        (x_train, y_train), (x_test, y_test) = dataset()
        if dataset == mnist.load_data or dataset == fashion_mnist.load_data:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
        y_train = to_categorical(y_train, num_classes=selected_dataset["nclass"])
        y_test = to_categorical(y_test, num_classes=selected_dataset["nclass"])
    else:
        ds = dataset()
        data = ds["raw"]["data"]
        label = ds['raw']["label"]
        #dividimos el dataset para crear train (2/3) y test (1/3)
        x_train = data[:2 * int(len(data) / 3)]
        x_test = data[2 * int(len(data) / 3):]
        mean_data = x_train.mean(axis=0)
        std_data = x_train.std(axis=0) + 1e-8
        x_train = (x_train - mean_data)/std_data
        x_test = (x_test - mean_data)/std_data
        y_train = to_categorical(label[:2 * int(len(label) / 3)], num_classes=selected_dataset["nclass"])
        y_test = to_categorical(label[2 * int(len(label) / 3):], num_classes=selected_dataset["nclass"])

    ## LOAD MODEL AND COMPILE IT (NEVER FORGET TO COMPILE!)
    model = model_fun(input_shape=x_train.shape[1:], nclasses=selected_dataset["nclass"], regularization=5e-4)
    model.compile(optimizer=optimizers.SGD(), metrics=['acc'], loss=loss)
    # model.fit(
    #    x_train, y_train, batch_size=batch_size, validation_data=(x_test, y_test), verbose=2
    # )

    ## LOAD E2EFS AND RUN IT
    fs_class = models.E2EFSSoft(n_features_to_select=selected_dataset["nfeat"]).attach(model).fit(
        x_train, y_train, batch_size=selected_dataset["batch"], validation_data=(x_test, y_test), verbose=2
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

    fs_class.fine_tuning(x_train, y_train, epochs=epochs, batch_size=selected_dataset["batch"], validation_data=(x_test, y_test),
                         callbacks=[LearningRateScheduler(scheduler)], verbose=2)
    print('FEATURE_RANKING :', fs_class.get_ranking())
    acc = fs_class.get_model().evaluate(x_test, y_test, batch_size=selected_dataset["batch"])[-1]
    print('ACCURACY : ', acc)
    nnz = np.count_nonzero(fs_class.get_mask())
    print('FEATURE_MASK NNZ :', nnz)

    tracker.stop()

    df = pd.read_csv(outputFileName)
    df["accuracy"] = acc
    df["feature_mask"] = nnz
    df.to_csv(outputFileName, index=False)