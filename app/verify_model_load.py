import os
import tensorflow as tf
from keras.src.initializers.random_initializers import GlorotUniform

original_init = GlorotUniform.__init__

def compat_init(self, *args, **kwargs):
    kwargs.pop('input_axes', None)
    kwargs.pop('output_axes', None)
    return original_init(self, *args, **kwargs)

GlorotUniform.__init__ = compat_init
m = tf.keras.models.load_model('custom.keras', compile=False)
print(type(m))
print(m.input_shape)

