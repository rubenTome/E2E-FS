import keras
from keras import backend as K, ops, losses
# from keras import mixed_precision

#wrn164 con precision float64 no funciona
precision = "float16"
script = "/home/lidia/Documents/ruben/E2E-FS/example.py"
outputFileName = "emissions_" + precision + ".csv"
n_features_to_select = 6266 #mnist=39 colon=1000 leukemia=1785 lung181=6266 lymphoma=2013
nclasses = 2 #mnist=10, microarray=2
loss = losses.CategoricalCrossentropy(from_logits=True) #"binary_crossentropy" #mnist="categorical_crossentropy" microarray="binary_crossentropy"
batch_size = 16 #mnist=128 microarray=2
model_fun = "three_layer_nn" #posibles valores="wrn164","three_layer_nn"
dataset = "lung181" #posibles valores="colon","leukemia","lung181","lymphoma","mnist" 

print('using precision:', precision, 'ok')
keras.config.set_floatx(precision)
# mixed_precision.set_global_policy('mixed_float16')
if precision == "float16":
    K.set_epsilon(1e-2)
# if precision == "float64":
#     K.set_epsilon(2.220446049250313e-16)

bcknd = K.backend()
ops.epsilon = lambda : 1e-2 if precision == 'float16' else 1e-8