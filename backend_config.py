# import keras
# from keras import backend as K, ops, losses

# mnist = {"name": "mnist", "nfeat": 39, "nclass": 10, "batch": 128, "model": "wrn164", "epochs": 60}
# fashion_mnist = {"name": "fashion_mnist", "nfeat": 39, "nclass": 10, "batch": 128, "model": "wrn164","epochs": 60}
# eurosat = {"name": "eurosat", "nfeat": 2048, "nclass": 10, "batch": 128, "model": "wrn164","epochs": 60}
# colorectal_histology = {"name": "colorectal_histology", "nfeat": 33750, "nclass": 8, "batch": 128, "model": "wrn164","epochs": 60}
# cifar10 = {"name": "cifar10", "nfeat": 512, "nclass": 10, "batch": 128, "model": "wrn164","epochs": 60}

# colon = {"name": "colon", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
# leukemia = {"name": "leukemia", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
# lung181 = {"name": "lung181", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
# lymphoma = {"name": "lymphoma", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}

# gisette = {"name": "gisette", "nfeat": 10, "nclass": 2, "batch": 128, "model": "linearSVC","epochs": 150150}
# dexter = {"name": "dexter", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
# gina = {"name": "gina", "nfeat": 10, "nclass": 2, "batch": 16, "model": "linearSVC","epochs": 150}
# madelon = {"name": "madelon", "nfeat": 5, "nclass": 2, "batch": 16, "model": "three_layer_nn","epochs": 150}

# precision = "float16"
# script = "/home/lidia/Documents/ruben/E2E-FS/example.py"
# selected_dataset = gina
# loss = losses.CategoricalCrossentropy(from_logits=False)
# outputFileName = "emissions_" + selected_dataset["name"] + "_" + precision + ".csv"

# print('using precision:', precision, 'ok')
# keras.config.set_floatx(precision)
# if precision == "float16":
#     K.set_epsilon(1e-2)

# bcknd = K.backend()
# ops.epsilon = lambda : 1e-2 if precision == 'float16' else 1e-8
