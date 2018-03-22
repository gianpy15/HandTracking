from keras.engine import Layer
from keras import backend as K


class Abs(Layer):
    def __init__(self, **kwargs):
        super(Abs, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        return K.abs(x)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)