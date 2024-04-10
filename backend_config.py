import keras
from keras import backend as K

precision = "float32"
script = "/home/lidia/Documents/ruben/E2E-FS/example.py"
outputFileName = "emissions_" + precision + ".csv"

K.set_floatx(precision)
if precision == "float16":
    K.set_epsilon(1/1024)
if precision == "float64":
    K.set_epsilon(2.220446049250313e-16)

bcknd = K.backend()