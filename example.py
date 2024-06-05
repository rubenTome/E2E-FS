from codecarbon import EmissionsTracker
from backend_config import outputFileName
tracker = EmissionsTracker(log_level="warning", output_file=outputFileName)
tracker.start()
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import time
import numpy as np
import pandas as pd
import keras
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras.datasets import mnist, cifar10, fashion_mnist
from keras import optimizers, backend as K, callbacks
from e2efs import models, e2efs_layers_tf216 as e2efs_layers, callbacks as clbks, optimizers_tf216
from backend_config import bcknd, ops, selected_dataset, loss
from src.wrn.network_models import wrn164, three_layer_nn, three_layer_nn_v2
from dataset_reader import colon, leukemia, lung181, lymphoma, gisette, dexter, gina, madelon
from src.svc.models import LinearSVC
import tensorflow_datasets as tfds
#import warnings
#warnings.filterwarnings("error", category=RuntimeWarning)

ops.cast_to_floatx = lambda x: ops.cast(x, keras.config.floatx())
K.backend = bcknd

#params for fs_challenge datasets
mu = 100
kernel = 'linear'
reps = 1
verbose = 0
loss_function = 'square_hinge'
optimizer_class = optimizers_tf216.E2EFS_Adam
initial_lr = .01

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
def train_Keras(train_X, train_y, test_X, test_y, normalization_func, kwargs, e2efs_class=None, n_features=None, epochs=150):
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

print("model function:", selected_dataset["model"])
if selected_dataset["model"] == "three_layer_nn":
    model_fun = three_layer_nn
elif selected_dataset["model"] == "three_layer_nn_v2":
    model_fun = three_layer_nn_v2
elif selected_dataset["model"] == "wrn164":
    model_fun = wrn164
elif selected_dataset["model"] == "linearSVC":
    model_fun = LinearSVC
else:
    raise Exception("invalid model function")

microarr = False
tf_ds = False
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
elif selected_dataset["name"] == "gisette":
    microarr = True #TEMPORAL
    dataset = gisette.load_dataset
    normalization_func = gisette.Normalize
elif selected_dataset["name"] == "dexter":
    microarr = True
    dataset = dexter.load_dataset
    normalization_func = dexter.Normalize
elif selected_dataset["name"] == "gina":
    microarr = True
    dataset = gina.load_dataset
    normalization_func = gina.Normalize
elif selected_dataset["name"] == "madelon":
    microarr = True
    dataset = madelon.load_dataset
    normalization_func = madelon.Normalize
elif selected_dataset["name"] == "eurosat" or selected_dataset["name"] == "colorectal_histology":#faltan añadir mas ds de tensorflow
    tf_ds = True
else:
    raise Exception("Invalid dataset", selected_dataset["name"])

if __name__ == '__main__':

    ## LOAD DATA
    # if temporal, para diferenciar microarray de los demas conjuntos de datos
    if microarr:
        ds = dataset()
        data = ds["raw"]["data"]
        label = ds["raw"]["label"]
        #dividimos el dataset para crear train (2/3) y test (1/3)
        x_train = data[:2 * int(len(data) / 3)]
        x_test = data[2 * int(len(data) / 3):]
        mean_data = x_train.mean(axis=0)
        std_data = x_train.std(axis=0) + 1e-8
        x_train = (x_train - mean_data) / std_data
        x_test = (x_test - mean_data) / std_data
        y_train = to_categorical(label[:2 * int(len(label) / 3)], num_classes=selected_dataset["nclass"])
        y_test = to_categorical(label[2 * int(len(label) / 3):], num_classes=selected_dataset["nclass"])
    elif tf_ds:
        x_train = []
        y_train = []
        x_test = []
        y_test = []
        if selected_dataset["name"] == "eurosat" or selected_dataset["name"] == "colorectal_histology":
            train_ds, test_ds = tfds.load(selected_dataset["name"], as_supervised=True, split=["train[:70%]", "train[70%:]"])
        else:
            train_ds, test_ds = tfds.load(selected_dataset["name"], as_supervised=True, split=["train", "test"])
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
        if dataset == mnist.load_data or dataset == fashion_mnist.load_data:
            x_train = np.expand_dims(x_train, axis=-1)
            x_test = np.expand_dims(x_test, axis=-1)
        y_train = to_categorical(y_train, num_classes=selected_dataset["nclass"])
        y_test = to_categorical(y_test, num_classes=selected_dataset["nclass"])

    ## LOAD MODEL AND COMPILE IT (NEVER FORGET TO COMPILE!)
    if selected_dataset["model"] == "linearSVC":
        model_kwargs = {'mu': mu / len(x_train), 'kernel': kernel, 'degree': 3}
        model = train_Keras(x_train, y_train, x_test, y_test, normalization_func, model_kwargs, 
                            e2efs_class=e2efs_layers.E2EFSSoft, n_features=selected_dataset["nfeat"])
    else:
        model = model_fun(input_shape=x_train.shape[1:], nclasses=selected_dataset["nclass"], regularization=5e-4)
        model.compile(optimizer=optimizers.SGD(), metrics=['acc'], loss=loss)

    ## LOAD E2EFS AND RUN IT
    fs_class = models.E2EFSSoft(n_features_to_select=selected_dataset["nfeat"]).attach(model).fit(
        x_train, y_train, batch_size=selected_dataset["batch"], validation_data=(x_test, y_test), verbose=2
    )

    ## FINE TUNING
    fs_class.fine_tuning(x_train, y_train, epochs=selected_dataset["epochs"], batch_size=selected_dataset["batch"], validation_data=(x_test, y_test),
                         callbacks=[LearningRateScheduler(scheduler_ft)], verbose=2)
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