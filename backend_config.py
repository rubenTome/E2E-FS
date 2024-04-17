import keras
from keras import backend as K, ops
# from keras import mixed_precision

precision = "float16"
script = "/home/lidia/Documents/ruben/E2E-FS/example.py"
outputFileName = "emissions_" + precision + ".csv"
n_features_to_select = 2013 #mnist=39 colon=1000 leukemia=1785 lung181=6266 lymphoma=2013
nclasses = 2 #mnist=10
loss = "binary_crossentropy" #mnist="binary_crossentropy"
batch_size = 2 #mnist=128

print('using precision:', precision, 'ok')
keras.config.set_floatx(precision)
# mixed_precision.set_global_policy('mixed_float16')
# if precision == "float16":
#     K.set_epsilon(1/1024)
# if precision == "float64":
#     K.set_epsilon(2.220446049250313e-16)

bcknd = K.backend()
ops.epsilon = K.epsilon