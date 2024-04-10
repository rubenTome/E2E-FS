from codecarbon import EmissionsTracker
csvFile = "emissions_mp.csv"
tracker = EmissionsTracker(log_level="warning", output_file=csvFile)
tracker.start()
from keras.datasets import mnist
from keras.callbacks import LearningRateScheduler
from keras.utils import to_categorical
from keras import optimizers, mixed_precision
from e2efs import models
from src.wrn.network_models import three_layer_nn_q
import numpy as np
from keras.mixed_precision import LossScaleOptimizer
import pandas as pd


mixed_precision.set_global_policy('mixed_float16')

if __name__ == '__main__':

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    model = three_layer_nn_q(input_shape=x_train.shape[1:], nclasses=10, regularization=5e-4, layer_dims=[50, 25, 10])
    model.compile(optimizer=LossScaleOptimizer(inner_optimizer=optimizers.SGD()), metrics=['acc'], loss='categorical_crossentropy')

    fs_class = models.E2EFSSoft(n_features_to_select=39).attach(model).fit(
        x_train, y_train, batch_size=128, validation_data=(x_test, y_test), verbose=2
    )
    
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

df = pd.read_csv(csvFile)
df["accuracy"] = acc
df["feature_mask"] = nnz
df.to_csv(csvFile, index=False)