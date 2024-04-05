import keras
from keras import backend as K

floatx = "float32"

K.set_floatx(floatx)
if floatx == "float16":
    K.set_epsilon(1/1024)
if floatx == "float64":
    K.set_epsilon(2.220446049250313e-16)

bcknd = K.backend()