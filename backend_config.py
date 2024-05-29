import keras
from keras import backend as K, ops, losses

#wrn164 con precision float64 no funciona
#wrn164 con precision float16 loss infinito

#POSIBLES VALORES PARA MODEL: "wrn164","three_layer_nn","three_layer_nn_v2"
mnist = {"name": "mnist", "nfeat": 392, "nclass": 10, "batch": 128, "model": "wrn164"}
fashion_mnist = {"name": "fashion_mnist", "nfeat": 392, "nclass": 10, "batch": 128, "model": "wrn164"}
cifar10 = {"name": "cifar10", "nfeat": 512, "nclass": 10, "batch": 128, "model": "wrn164"}
colon = {"name": "colon", "nfeat": 1000, "nclass": 2, "batch": 16, "model": "three_layer_nn_v2"}
leukemia = {"name": "leukemia", "nfeat": 1785, "nclass": 2, "batch": 16, "model": "three_layer_nn"}
lung181 = {"name": "lung181", "nfeat": 6266, "nclass": 2, "batch": 16, "model": "three_layer_nn"}
lymphoma = {"name": "lymphoma", "nfeat": 2013, "nclass": 2, "batch": 16, "model": "three_layer_nn"}

precision = "float16"
script = "/home/lidia/Documents/ruben/E2E-FS/example.py"
selected_dataset = mnist
loss = losses.CategoricalCrossentropy(from_logits=False)
epochs = 60
outputFileName = "emissions_" + selected_dataset["name"] + "_" + precision + ".csv"

print('using precision:', precision, 'ok')
keras.config.set_floatx(precision)
if precision == "float16":
    K.set_epsilon(1e-2)

bcknd = K.backend()
ops.epsilon = lambda : 1e-2 if precision == 'float16' else 1e-8