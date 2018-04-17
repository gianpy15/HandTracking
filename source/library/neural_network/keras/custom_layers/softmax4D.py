from keras.engine import Layer
from keras import backend as K


class Softmax4D(Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)


class MultiLayerSoftmax4D(Layer):
    def __init__(self, axis=(-1,), **kwargs):
        self.axis = axis
        super(MultiLayerSoftmax4D, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        result = 0
        for x in self.axis:
            e = K.exp(x - K.max(x, axis=x, keepdims=True))
            s = K.sum(e, axis=x, keepdims=True)
            result += e / s
        return result

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_output_shape_for(self, input_shape):
        return self.compute_output_shape(input_shape)
