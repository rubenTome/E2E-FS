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
import tensorflow_datasets as tfds
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

if __name__ == '__main__':

    ## LOAD DATA
    print("used dataset:", selected_dataset["name"])
    if selected_dataset["name"] == "mnist":
        train_ds, test_ds = tfds.load('mnist', as_supervised=True, split=["train", "test"])
        x_train = []
        y_train = []
        x_test = []
        y_test = []
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
        raise Exception("Invalid dataset", selected_dataset["name"])

    ## LOAD MODEL AND COMPILE IT (NEVER FORGET TO COMPILE!)
    model = model_fun(input_shape=x_train.shape[1:], nclasses=selected_dataset["nclass"], regularization=5e-4)
    model.compile(optimizer=optimizers.SGD(), metrics=['acc'], loss=loss)

    ## LOAD E2EFS AND RUN IT
    fs_class = models.E2EFSSoft(n_features_to_select=selected_dataset["nfeat"]).attach(model).fit(
        x_train, y_train, batch_size=selected_dataset["batch"], validation_data=(x_test, y_test), verbose=2
    )

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