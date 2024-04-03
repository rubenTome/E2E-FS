from keras import backend as K

floatx = "float32"

K.set_floatx(floatx)
if floatx == "float16":
    K.set_epsilon(1/1024)
bcknd = K.backend()