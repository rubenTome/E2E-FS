import keras
from keras import backend as K, ops
# from keras import mixed_precision

precision = "float32"
script = "/home/lidia/Documents/ruben/E2E-FS/example.py"
outputFileName = "emissions_" + precision + ".csv"

print('using precision:', precision, 'ok')
keras.config.set_floatx(precision)
# mixed_precision.set_global_policy('mixed_float16')
# if precision == "float16":
#     K.set_epsilon(1/1024)
# if precision == "float64":
#     K.set_epsilon(2.220446049250313e-16)

bcknd = K.backend()
ops.epsilon = K.epsilon