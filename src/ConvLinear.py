from keras import layers
import tensorflow as tf

class ConvLinear(layers.Layer):
 
    def __init__(self, units, **kwargs):
        super(ConvLinear, self).__init__(**kwargs)         
        self.units = units     
        
    def build(self, input_shape):
        # The kernel is a 1D convolutional filter of size 1 with 'units' filters        
        self.kernel = self.add_weight(
            name='kernel',
            shape=(1, input_shape[-1], self.units), # Shape: (kernel_size, input_channels, output_units)
            initializer='glorot_uniform',  # You can use different initializers
            trainable=True)
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros', # Bias initialization
            trainable=True)
        super(ConvLinear, self).build(input_shape)
 
    def call(self, inputs):
        # Reshape input to (batch_size, input_dim, 1)
        x = tf.expand_dims(inputs, axis=1)# (batch_size, 1, input_dim)
        # Apply Conv1D with kernel_size=1
        x = tf.nn.conv1d(x, self.kernel, stride=1, padding="SAME")
        # Add bias to the result
        x = x + self.bias # Broadcasting the bias to the result
        # Squeeze the last dimension to get (batch_size, units)
        return tf.squeeze(x, axis=1)