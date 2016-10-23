from keras.engine.topology import Layer

from keras import backend as K

class Denormalize(Layer):
    '''
    Custom layer to denormalize the final Convolution layer activations (tanh)

    Since tanh scales the output to the range (-1, 1), we add 1 to bring it to the
    range (0, 2). We then multiply it by 127.5 to scale the values to the range (0, 255)
    '''

    def __init__(self, **kwargs):
        super(Denormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Scales the tanh output activations from previous layer (-1, 1) to the
        range (0, 255)
        '''

        return (x + 1) * 127.5

    def get_output_shape_for(self, input_shape):
        return input_shape


class VGGNormalize(Layer):
    '''
    Custom layer to subtract the outputs of previous layer by 120,
    to normalize the inputs to the VGG network.
    '''

    def __init__(self, **kwargs):
        super(VGGNormalize, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        '''
        Since individual channels cannot be altered in a TensorVariable, therefore
        we subtract it by 120, similar to the chainer implementation.
        '''
        if K.backend() == "theano":
            import theano.tensor as T
            T.set_subtensor(x[:, 0, :, :], x[:, 0, :, :] - 103.939, inplace=True)
            T.set_subtensor(x[:, 1, :, :], x[:, 1, :, :] - 116.779, inplace=True)
            T.set_subtensor(x[:, 2, :, :], x[:, 2, :, :] - 123.680, inplace=True)
        else:
            # No exact substitute for set_subtensor in tensorflow
            # So we subtract an approximate value
            x = x - 120
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


class ReflectionPadding2D(Layer):
    ''' Mirror padding for 2 dimensional images.

        Mirror padding is used to apply a 2D convolution avoiding the border
        effects that one normally gets with zero padding.
        We assume that the filter has an odd size.
        To obtain a filtered tensor with the same output size, substitute
        a ``conv2d(images, filters, mode="half")`` with
        ``conv2d(mirror_padding(images, filters.shape), filters, mode="valid")``.

        Parameters
        ----------
        images : Tensor
            4D tensor containing a set of images.
        filter_size : tuple
            Spatial size of the filter (height, width).

        Returns
        -------
        padded : Tensor
            4D tensor containing the padded set of images.
    '''

    def __init__(self, padding=(1, 1), dim_ordering='default', **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)

        assert K.backend() == "theano", "ReflectionPadding is supported on Theano only for the moment"

        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.padding = tuple(padding)

        if (self.padding[0] % 2) == 0 or (self.padding[1] % 2) == 0:
            raise ValueError('Padding size must be an odd number')

        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.dim_ordering = dim_ordering

    def get_output_shape_for(self, input_shape):
        if self.dim_ordering == 'th':
            width = input_shape[2] + 2 * self.padding[0] if input_shape[2] is not None else None
            height = input_shape[3] + 2 * self.padding[1] if input_shape[3] is not None else None
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height)
        elif self.dim_ordering == 'tf':
            width = input_shape[1] + 2 * self.padding[0] if input_shape[1] is not None else None
            height = input_shape[2] + 2 * self.padding[1] if input_shape[2] is not None else None
            return (input_shape[0],
                    width,
                    height,
                    input_shape[3])
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        input_shape = x.shape

        if self.dim_ordering == "th":
            b, k, r, c = input_shape
        else:
            b, r, c, k = input_shape

        if k % 2 == 0:
            raise ValueError("Number of filters must be an odd number")

        w_pad = self.padding[0]
        h_pad = self.padding[1]

        from theano.tensor import set_subtensor, zeros

        out = zeros(self.get_output_shape_for(input_shape))

        # Copy the original image to the central part.

        # TODO: Make this compatible with Tensorflow
        out = set_subtensor(out[:, :, w_pad:w_pad + r, h_pad:c + h_pad], x)

        # Copy borders.

        # Note that we don't mirror starting at pixel number 0: assuming that
        # we have a symmetric, odd filter, the central element of the filter
        # will run along the original border, and we need to match the
        # statistics of the elements around it.

        out = set_subtensor(out[:, :, w_pad:-w_pad, :h_pad], x[:, :, :, h_pad:0:-1])
        out = set_subtensor(out[:, :, w_pad:-w_pad, -h_pad:], x[:, :, :, -2:-h_pad - 2:-1])
        out = set_subtensor(out[:, :, :w_pad, :], out[:, :, 2 * w_pad:w_pad:-1, :])
        out = set_subtensor(out[:, :, -w_pad:, :], out[:, :, -w_pad - 2:-2 * w_pad - 2:-1, :])

        return out

    def get_config(self):
        config = {'padding': self.padding}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))