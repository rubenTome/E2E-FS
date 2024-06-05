import keras
from keras import backend as K, ops, losses

mnist = {"name": "mnist", "nfeat": 39, "nclass": 10, "batch": 128, "model": "wrn164", "epochs": 60}
fashion_mnist = {"name": "fashion_mnist", "nfeat": 39, "nclass": 10, "batch": 128, "model": "wrn164","epochs": 60}
eurosat = {"name": "eurosat", "nfeat": 2048, "nclass": 10, "batch": 128, "model": "wrn164","epochs": 60}
colorectal_histology = {"name": "colorectal_histology", "nfeat": 33750, "nclass": 8, "batch": 128, "model": "wrn164","epochs": 60}
cifar10 = {"name": "cifar10", "nfeat": 512, "nclass": 10, "batch": 128, "model": "wrn164","epochs": 60}
colon = {"name": "colon", "nfeat": 1000, "nclass": 2, "batch": 16, "model": "three_layer_nn","epochs": 60}
leukemia = {"name": "leukemia", "nfeat": 1785, "nclass": 2, "batch": 16, "model": "three_layer_nn","epochs": 60}
lung181 = {"name": "lung181", "nfeat": 6266, "nclass": 2, "batch": 16, "model": "three_layer_nn","epochs": 60}
lymphoma = {"name": "lymphoma", "nfeat": 2013, "nclass": 2, "batch": 16, "model": "three_layer_nn","epochs": 60}
gisette = {"name": "gisette", "nfeat": 20, "nclass": 2, "batch": 128, "model": "linearSVC","epochs": 100}
dexter = {"name": "dexter", "nfeat": 9947, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
gina = {"name": "gina", "nfeat": 485, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
madelon = {"name": "madelon", "nfeat": 30, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}

precision = "float32"
script = "/home/lidia/Documents/ruben/E2E-FS/example.py"
selected_dataset = colon
loss = losses.CategoricalCrossentropy(from_logits=False)
outputFileName = "emissions_" + selected_dataset["name"] + "_" + precision + ".csv"

print('using precision:', precision, 'ok')
keras.config.set_floatx(precision)
if precision == "float16":
    K.set_epsilon(1e-2)

bcknd = K.backend()
ops.epsilon = lambda : 1e-2 if precision == 'float16' else 1e-8
