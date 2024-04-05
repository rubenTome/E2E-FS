import keras
from keras import backend as K

floatx = "float16"

K.set_floatx(floatx)
if floatx == "float16":
    K.set_epsilon(1/1024)
bcknd = K.backend()
