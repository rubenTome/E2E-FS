from codecarbon import EmissionsTracker
import os
import time
import numpy as np
import pandas as pd
import keras
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras.datasets import mnist as mn, cifar10 as cf, fashion_mnist as fs
from keras import optimizers, backend as K, callbacks, ops, losses
from e2efs import models, e2efs_layers_tf216 as e2efs_layers, callbacks as clbks, optimizers_tf216
#from src.network_models import three_layer_nn
from src.wrn.network_models import wrn164, densenet, three_layer_nn, three_layer_nn_v2
from dataset_reader import colon as cl, leukemia as lk, lung181 as ln, lymphoma as lm, gisette as gs, dexter as dx, gina as gn, madelon as md
from src.svc.models import LinearSVC
import tensorflow_datasets as tfds
import tensorflow
from scipy.special import erf

class Normalize_KDS:
    def __init__(self):
        self.stats = None
    def fit(self, X):
        mean = np.mean(X, axis=0)
        std = np.sqrt(np.square(X - mean).sum(axis=0) / max(1, len(X) - 1))
        self.stats = (mean, std)
    def transform(self, X):
        transformed_X = erf((X - self.stats[0]) / (np.maximum(1e-6, self.stats[1]) * np.sqrt(2.)))
        return transformed_X
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

def scheduler_ft(epoch):
    if epoch < 20:
        return .1
    elif epoch < 40:
        return .02
    elif epoch < 50:
        return .004
    else:
        return .0008

def scheduler():
    def sch(epoch):
        if epoch < 50:
            return initial_lr
        elif epoch < 100:
            return .2 * initial_lr
        else:
            return .04 * initial_lr
    return sch

#model creation for linearSVC
def train_Keras_linearSVC(train_X, train_y, test_X, test_y, normalization_func, kwargs, e2efs_class=None, n_features=None, epochs=150):
    normalization = normalization_func()
    num_classes = train_y.shape[-1]

    norm_train_X = normalization.fit_transform(train_X)
    norm_test_X = normalization.transform(test_X)

    batch_size = max(2, len(train_X) // 50)
    class_weight = train_y.shape[0] / np.sum(train_y, axis=0)
    class_weight = num_classes * class_weight / class_weight.sum()
    sample_weight = None
    print('mu :', kwargs['mu'], ', batch_size :', batch_size, ', n_feats :', n_features)
    print('reps : ', reps, ', weights : ', class_weight)
    if num_classes == 2:
        sample_weight = np.zeros((len(norm_train_X),))
        sample_weight[train_y[:, 1] == 1] = class_weight[1]
        sample_weight[train_y[:, 1] == 0] = class_weight[0]
        class_weight = None

    svc_model = LinearSVC(nfeatures=norm_train_X.shape[1:], **kwargs)
    svc_model.create_keras_model(nclasses=num_classes)

    model_clbks = [
        callbacks.LearningRateScheduler(scheduler()),
    ]

    fs_callbacks = []

    if e2efs_class is not None:
        classifier = svc_model.model
        e2efs_layer = e2efs_class(n_features, input_shape=norm_train_X.shape[1:])
        model = e2efs_layer.add_to_model(classifier, input_shape=norm_train_X.shape[1:])
        fs_callbacks.append(
            clbks.E2EFSCallback(verbose=verbose)
        )
    else:
        model = svc_model.model
        e2efs_layer = None

    optimizer = optimizer_class(e2efs_layer )

    model.compile(
        loss=LinearSVC.loss_function(loss_function, class_weight),
        optimizer=optimizer,
        metrics=[LinearSVC.accuracy]
    )

    if e2efs_class is not None:
        model.fs_layer = e2efs_layer
        model.heatmap = e2efs_layer.moving_heatmap

        start_time = time.process_time()
        model.fit(
            norm_train_X, train_y, batch_size=batch_size,
            epochs=200000,
            callbacks=fs_callbacks,
            validation_data=(norm_test_X, test_y),
            class_weight=class_weight,
            sample_weight=sample_weight,
            verbose=verbose
        )
        model.fs_time = time.process_time() - start_time

    model.fit(
        norm_train_X, train_y, batch_size=batch_size,
        epochs=epochs,
        callbacks=model_clbks,
        validation_data=(norm_test_X, test_y),
        class_weight=class_weight,
        sample_weight=sample_weight,
        verbose=verbose
    )

    model.normalization = normalization

    return model

mnist = {"name": "mnist", "nfeat": 39, "nclass": 10, "batch": 128, "model": "three_layer_nn", "epochs": 60}
fashion_mnist = {"name": "fashion_mnist", "nfeat": 39, "nclass": 10, "batch": 128, "model": "three_layer_nn","epochs": 60}
eurosat = {"name": "eurosat", "nfeat": 2048, "nclass": 10, "batch": 128, "model": "wrn164","epochs": 60}
colorectal_histology = {"name": "colorectal_histology", "nfeat": 33750, "nclass": 8, "batch": 128, "model": "wrn164","epochs": 60}
cifar10 = {"name": "cifar10", "nfeat": 512, "nclass": 10, "batch": 128, "model": "three_layer_nn","epochs": 60}
colon = {"name": "colon", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
leukemia = {"name": "leukemia", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
lung181 = {"name": "lung181", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
lymphoma = {"name": "lymphoma", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
gisette = {"name": "gisette", "nfeat": 10, "nclass": 2, "batch": 128, "model": "linearSVC","epochs": 150}
dexter = {"name": "dexter", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
gina = {"name": "gina", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
madelon = {"name": "madelon", "nfeat": 5, "nclass": 2, "batch": 16, "model": "three_layer_nn","epochs": 150}

#SELECTED DATASETS AND PRECISIONS
datasets = [mnist, fashion_mnist, cifar10]#, eurosat, colorectal_histology, colon, leukemia, lung181, lymphoma, gisette, dexter, gina, madelon]
#SOLO USAR 1 PRECISION POR EJECUCION
precisions = ["float16"]

#params for train_Keras_XXX
mu = 100
kernel = 'linear'
reps = 1
verbose = 0
loss_function = 'square_hinge'
optimizer_class = optimizers_tf216.E2EFS_Adam
initial_lr = .01
regularization = 1e-3

#main loop
#if __name__ == '__main__':
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
loss = losses.CategoricalCrossentropy(from_logits=False)
print("Num GPUs Available: ", len(tensorflow.config.experimental.list_physical_devices('GPU')))
for ds in datasets:
    for prec in precisions:
        print('using precision:', prec, 'ok')
        outputFileName = "emissions_" + ds["name"] + "_" + prec + ".csv"
        tracker = EmissionsTracker(log_level="warning", output_file= "results/" + outputFileName)
        tracker.start()
        keras.config.set_floatx(prec)
        if prec == "float16":
            K.set_epsilon(1e-2)
        else:
            K.set_epsilon(1e-8)
        ops.epsilon = lambda : 1e-2 if prec == 'float16' else 1e-8
        ops.cast_to_floatx = lambda x: ops.cast(x, keras.config.floatx())
        microarr = False
        tf_ds = False

        print("model function:", ds["model"])
        if ds["model"] == "three_layer_nn":
            model_fun = three_layer_nn
        elif ds["model"] == "wrn164":
            model_fun = wrn164
        elif ds["model"] == "linearSVC":
            model_fun = LinearSVC
        elif ds["model"] == "densenet":
            model_fun = densenet
        else:
            raise Exception("invalid model function")

        print("used dataset:", ds["name"])
        if ds["name"] == "mnist":
            dataset = mn.load_data
            normalization_func = Normalize_KDS
        elif ds["name"] == "cifar10":
            dataset = cf.load_data
        elif ds["name"] == "fashion_mnist":
            dataset = fs.load_data
        elif ds["name"] == "colon":
            microarr = True
            dataset = cl.load_dataset
            normalization_func = cl.Normalize
        elif ds["name"] == "leukemia":
            microarr = True
            dataset = lk.load_dataset
            normalization_func = lk.Normalize
        elif ds["name"] == "lung181":
            microarr = True
            dataset = ln.load_dataset
            normalization_func = ln.Normalize
        elif ds["name"] == "lymphoma":
            microarr = True
            dataset = lm.load_dataset
            normalization_func = lm.Normalize
        elif ds["name"] == "gisette":
            microarr = True
            dataset = gs.load_dataset
            normalization_func = gs.Normalize
        elif ds["name"] == "dexter":
            microarr = True
            dataset = dx.load_dataset
            normalization_func = dx.Normalize
        elif ds["name"] == "gina":
            microarr = True
            dataset = gn.load_dataset
            normalization_func = gn.Normalize
        elif ds["name"] == "madelon":
            microarr = True
            dataset = md.load_dataset
            normalization_func = md.Normalize
        elif ds["name"] == "eurosat" or ds["name"] == "colorectal_histology":#faltan añadir mas ds de tensorflow
            tf_ds = True
        else:
            raise Exception("Invalid dataset", ds["name"])

        ## LOAD DATA
        # if temporal, para diferenciar microarray de los demas conjuntos de datos
        if microarr:
            dataset_loaded = dataset()
            data = dataset_loaded["raw"]["data"]
            label = dataset_loaded["raw"]["label"]
            #dividimos el dataset para crear train (2/3) y test (1/3)
            x_train = data[:2 * int(len(data) / 3)]
            x_test = data[2 * int(len(data) / 3):]
            mean_data = x_train.mean(axis=0)
            std_data = x_train.std(axis=0) + 1e-8
            x_train = (x_train - mean_data) / std_data
            x_test = (x_test - mean_data) / std_data
            y_train = to_categorical(label[:2 * int(len(label) / 3)], num_classes=ds["nclass"])
            y_test = to_categorical(label[2 * int(len(label) / 3):], num_classes=ds["nclass"])
        elif tf_ds:
            x_train = []
            y_train = []
            x_test = []
            y_test = []
            if ds["name"] == "eurosat" or ds["name"] == "colorectal_histology":
                train_ds, test_ds = tfds.load(ds["name"], as_supervised=True, split=["train[:70%]", "train[70%:]"])
            else:
                train_ds, test_ds = tfds.load(ds["name"], as_supervised=True, split=["train", "test"])
            for image, label in tfds.as_numpy(train_ds):
                x_train.append(image)
                y_train.append(label)
            for image, label in tfds.as_numpy(test_ds):
                x_test.append(image)
                y_test.append(label)
            x_train = np.array(x_train)
            x_test = np.array(x_test)
            y_train = to_categorical(np.array(y_train).astype(float))
            y_test = to_categorical(np.array(y_test).astype(float))
        else:
            (x_train, y_train), (x_test, y_test) = dataset()
            if dataset == mn.load_data or dataset == fs.load_data:
                x_train = np.expand_dims(x_train, axis=-1)
                x_test = np.expand_dims(x_test, axis=-1)
            y_train = to_categorical(y_train, num_classes=ds["nclass"])
            y_test = to_categorical(y_test, num_classes=ds["nclass"])

        ## LOAD MODEL AND COMPILE IT (NEVER FORGET TO COMPILE!)
        if ds["model"] == "linearSVC":
            model_kwargs = {'mu': mu / len(x_train), 'kernel': kernel, 'degree': 3}
            model = train_Keras_linearSVC(x_train, y_train, x_test, y_test, normalization_func, model_kwargs,
                                e2efs_class=e2efs_layers.E2EFSSoft, n_features=ds["nfeat"])
        elif ds["model"] == "three_layer_nn":
            model = three_layer_nn_v2(input_shape=x_train.shape[1:], nclasses=ds["nclass"], regularization=regularization)
            model.compile(optimizer=optimizers.SGD(), metrics=['acc'], loss=loss)
        else:
            model = model_fun(input_shape=x_train.shape[1:], nclasses=ds["nclass"], regularization=5e-4)
            model.compile(optimizer=optimizers.SGD(), metrics=['acc'], loss=loss)
        print(model.summary())
        ## LOAD E2EFS AND RUN IT
        fs_class = models.E2EFSSoft(n_features_to_select=ds["nfeat"]).attach(model).fit(
            x_train, y_train, batch_size=ds["batch"], validation_data=(x_test, y_test), verbose=2
        )

        ## FINE TUNING
        fs_class.fine_tuning(x_train, y_train, epochs=ds["epochs"], batch_size=ds["batch"], validation_data=(x_test, y_test),
                            callbacks=[LearningRateScheduler(scheduler_ft)], verbose=2)
        print('FEATURE_RANKING :', fs_class.get_ranking())
        acc = fs_class.get_model().evaluate(x_test, y_test, batch_size=ds["batch"])[-1]
        print('ACCURACY : ', acc)
        nnz = np.count_nonzero(fs_class.get_mask())
        print('FEATURE_MASK NNZ :', nnz)

        tracker.stop()

        df = pd.read_csv("results/" + outputFileName)
        df["accuracy"] = acc
        df["feature_mask"] = nnz
        df.to_csv("results/" + outputFileName, index=False)