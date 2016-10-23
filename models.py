from loss import *
import img_utils
import h5py

from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge, BatchNormalization, Activation
from keras.layers.convolutional import Convolution2D, AveragePooling2D, MaxPooling2D, Deconvolution2D, Cropping2D
from keras.utils.data_utils import get_file

from layers import VGGNormalize, Denormalize, ReflectionPadding2D

# VGG-16 Weights Path
THEANO_WEIGHTS_PATH_NO_TOP = r'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_th_dim_ordering_th_kernels_notop.h5'
TENSORFLOW_WEIGHTS_PATH_NO_TOP = r"https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


def pooling_func(x, pooltype):
    '''
    Pooling function used in VGG

    Args:
        x: previous layer
        pooltype: int, 1 refers to AveragePooling2D. All other values refer to MaxPooling2D

    Returns:

    '''
    if pooltype == 1:
        return AveragePooling2D((2, 2), strides=(2, 2))(x)
    else:
        return MaxPooling2D((2, 2), strides=(2, 2))(x)


class VGG:
    '''
    Helper class to load VGG and its weights to the FastNet model
    '''

    def __init__(self, img_height=256, img_width=256):
        self.img_height = img_height
        self.img_width = img_width

    def append_vgg_model(self, model_input, x_in, pool_type=0):
        '''
        Adds the VGG model to the FastNet model. It concatenates the original input to the output generated
        by the FastNet model. This is used to compute output features of VGG for the input image.

        Then it rescales the FastNet outputs and initial input to range [-127.5, 127.5] with lambda layer
        Ideally, I would like to subtract the channel means individually, but that is not efficient.
        Therefore, the closest approximate is to scale the values in the range [-127.5, 127.5]

        After this it adds the VGG layers.

        Args:
            model_input: Input to the FastNet model
            x_in: Output of last layer of FastNet model
            pool_type: int, 1 = AveragePooling, otherwise uses MaxPooling

        Returns: Model (FastNet + VGG)

        '''

        if K.image_dim_ordering() == "th":
            true_X_input = Input(shape=(3, self.img_width, self.img_height))
        else:
            true_X_input = Input(shape=(self.img_width, self.img_height, 3))

        # Append the initial input to the FastNet input to the VGG inputs
        x = merge([x_in, true_X_input], mode='concat', concat_axis=0)

        # Normalize the inputs via custom VGG Normalization layer
        x = VGGNormalize(name="vgg_normalize")(x)

        # Begin adding the VGG layers
        x = Convolution2D(64, 3, 3, activation='relu', name='conv1_1')(x)
        x = Convolution2D(64, 3, 3, activation='relu', name='conv1_2', border_mode='same')(x)
        x = pooling_func(x, pool_type)

        x = Convolution2D(128, 3, 3, activation='relu', name='conv2_1', border_mode='same')(x)
        x = Convolution2D(128, 3, 3, activation='relu', name='conv2_2', border_mode='same')(x)
        x = pooling_func(x, pool_type)

        x = Convolution2D(256, 3, 3, activation='relu', name='conv3_1', border_mode='same')(x)
        x = Convolution2D(256, 3, 3, activation='relu', name='conv3_2', border_mode='same')(x)
        x = Convolution2D(256, 3, 3, activation='relu', name='conv3_3', border_mode='same')(x)
        x = pooling_func(x, pool_type)

        x = Convolution2D(512, 3, 3, activation='relu', name='conv4_1', border_mode='same')(x)
        x = Convolution2D(512, 3, 3, activation='relu', name='conv4_2', border_mode='same')(x)
        x = Convolution2D(512, 3, 3, activation='relu', name='conv4_3', border_mode='same')(x)
        x = pooling_func(x, pool_type)

        x = Convolution2D(512, 3, 3, activation='relu', name='conv5_1', border_mode='same')(x)
        x = Convolution2D(512, 3, 3, activation='relu', name='conv5_2', border_mode='same')(x)
        x = Convolution2D(512, 3, 3, activation='relu', name='conv5_3', border_mode='same')(x)
        x = pooling_func(x, pool_type)

        model = Model([model_input, true_X_input], x)

        # Loading VGG 16 weights
        if K.image_dim_ordering() == "th":
            weights_name = "vgg16_weights_th_dim_ordering_th_kernels_notop.h5"
            weights_path = THEANO_WEIGHTS_PATH_NO_TOP
        else:
            weights_name = "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"
            weights_path = TENSORFLOW_WEIGHTS_PATH_NO_TOP

        weights = get_file(weights_name, weights_path, cache_subdir='models')
        f = h5py.File(weights)

        layer_names = [name for name in f.attrs['layer_names']]

        for i, layer in enumerate(model.layers[-18:]):
            g = f[layer_names[i]]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)
        print('VGG Model weights loaded.')

        # Freeze all VGG layers
        for layer in model.layers[-19:]:
            layer.trainable = False

        self.model = model
        return model


class FastStyleNet:
    ''' Fast Style Network
    From the paper Perceptual losses for Real Time Style Transfer (http://arxiv.org/abs/1603.08155)
    '''

    def __init__(self, img_width=256, img_height=256, kernel_size=3, pool_type=0,
                 style_weight=1., content_weight=10., tv_weight=8.5e-5, model_width="thin",
                 model_depth="shallow", save_fastnet_model=None):
        '''
        Creates a FastStyleNet object which can be used to train, validate or predict networks

        This is used to create and manage the internal Keras Model so as to provide easy helper functions.

        Args:
            img_width: image width (im_rows for Keras)
            img_height: image_height (im_cols for Keras)

            kernel_size: kernel shape. Only 3x3 is tested for Theano backend. Tensorflow backend is not supported yet.
            pool_type: decides Max or Average Pooling. 1 is for AveragePooling, all other values default to MaxPooling

            style_weight: weight to style loss (author suggests 10.0)
            content_weight: weight to content loss (author suggests 1.0)
            tv_weight: weight to total variation loss (paper suggests any value between 1e-4 to 1e-6)

            model_width: Can take either "thin" or "wide".
                         With "thin", the internal nb_filters is 64.
                         With "wide", the internal nb_filters is 128.

                         "wide" increases the time required per iteration, but it also boosts training performance

            model_depth: Can take either "shallow" or "deep".
                         With "shallow", creates the original network in the paper.
                         With "deep", creates one more pooling and deconvolution layer. This improves training
                         speed and performance, but the style patches become very small.

                         It is advisable to use --image_size = 512 when using the deeper model.
                         This requires far more training time (0.65 seconds per iteration), however the results
                         are far better.

        '''

        self.img_width = img_width
        self.img_height = img_height
        self.k = kernel_size
        self.pool_type = pool_type

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

        self.features = 64 if model_width == "thin" else 128
        self.deep_model = False if model_depth == "shallow" else True

        self.vgg_style_func = None
        self.vgg_content_func = None
        self.fastnet_predict_func = None

        self.model = None  # Preserves the main training instance
        self.model_save_path = save_fastnet_model

    def create_model(self, style_name=None, train_mode=False, style_image_path=None, validation_path=None):
        '''
        Creates the FastNet model which can be used in train mode, predict mode and validation mode.

        If train_mode = True, this model appends the VGG model to the end of the FastNet model.
        In train mode, it requires style image path to be supplied.

        If train_mode = False and validation_path = None, this model is in predict mode.
        In predict mode, it requires a style_name to be supplied, whose weights it will try to load.

        If validation_path is not None, this model is in validation mode.
        In validation mode, it simply loads the weights provided by the validation_path and does not append VGG

        Args:
            style_name: Used in predict mode, used to load correct weights of the style
            train_mode: Used to activate train mode. Default is predict mode.
            style_image_path: Path to the style image. Necessary if in train mode.
            validation_path: Path to the validation weights that need to be loaded.

        Returns: FastNet Model (Prediction mode / Validation mode) or FastNet + VGG Model (Train mode)

        '''

        if train_mode and style_image_path is None:
            raise Exception('Style reference path must be supplied if training mode is enabled')

        self.mode = 2

        if K.image_dim_ordering() == "th":
            ip = Input(shape=(3, self.img_width, self.img_height), name="X_input")
        else:
            ip = Input(shape=(self.img_width, self.img_height, 3), name="X_input")

        #x = ReflectionPadding2D((41, 41))(ip)

        c1 = Convolution2D(32, 9, 9, activation='relu', border_mode='same', name='conv1')(ip)
        c1_b = BatchNormalization(axis=1, mode=self.mode, name="batchnorm1")(c1)

        c2 = Convolution2D(self.features, self.k, self.k, activation='relu', border_mode='same', subsample=(2, 2),
                           name='conv2')(c1_b)
        c2_b = BatchNormalization(axis=1, mode=self.mode, name="batchnorm2")(c2)

        c3 = Convolution2D(self.features, self.k, self.k, activation='relu', border_mode='same', subsample=(2, 2),
                           name='conv3')(c2_b)
        x = BatchNormalization(axis=1, mode=self.mode, name="batchnorm3")(c3)

        if self.deep_model:
            c4 = Convolution2D(self.features, self.k, self.k, activation='relu', border_mode='same', subsample=(2, 2),
                               name='conv4')(x)

            x = BatchNormalization(axis=1, mode=self.mode, name="batchnorm_4")(c4)

        r1 = self._residual_block(x, 1)
        r2 = self._residual_block(r1, 2)
        r3 = self._residual_block(r2, 3)
        r4 = self._residual_block(r3, 4)
        x = self._residual_block(r4, 5)

        if self.deep_model:
            d4 = Deconvolution2D(self.features, self.k, self.k, activation="relu", border_mode="same", subsample=(2, 2),
                                 output_shape=(1, self.features, self.img_width // 4, self.img_height // 4),
                                 name="deconv4")(x)

            x = BatchNormalization(axis=1, mode=self.mode, name="batchnorm_extra4")(d4)

        d3 = Deconvolution2D(self.features, self.k, self.k, activation="relu", border_mode="same", subsample=(2, 2),
                             output_shape=(1, self.features, self.img_width // 2, self.img_height // 2),
                             name="deconv3")(x)

        d3 = BatchNormalization(axis=1, mode=self.mode, name="batchnorm4")(d3)

        d2 = Deconvolution2D(self.features, self.k, self.k, activation="relu", border_mode="same", subsample=(2, 2),
                             output_shape=(1, self.features, self.img_width, self.img_height), name="deconv2")(d3)

        d2 = BatchNormalization(axis=1, mode=self.mode, name="batchnorm5")(d2)

        d1 = Convolution2D(3, 9, 9, activation='tanh', border_mode='same', name='valid')(d2)

        # Scale output to range [0, 255] via custom Denormalize layer
        d1 = Denormalize(name='fastnet_output')(d1)

        model = Model(ip, d1)

        if self.model_save_path is not None and self.model is None:
            model.save(self.model_save_path, overwrite=True)

        self.fastnet_outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
        fastnet_output_layer = model.layers[-1]

        if style_name is not None or validation_path is not None:
            try:
                if validation_path is not None:
                    path = validation_path
                else:
                    path = "weights/fastnet_%s.h5" % style_name

                model.load_weights(path)
                print('Fast Style Net weights loaded.')
            except:
                print('Weights for this style do not exist. Model weights not loaded.')

        # Add VGG layers to Fast Style Model
        if train_mode:
            model = VGG(self.img_height, self.img_width).append_vgg_model(model.input, x_in=model.output,
                                                                          pool_type=self.pool_type)

            if self.model is None:
                self.model = model

            self.vgg_output_dict = dict([(layer.name, layer.output) for layer in model.layers[-18:]])

            vgg_layers = dict([(layer.name, layer) for layer in model.layers[-18:]])

            style = img_utils.preprocess_image(style_image_path, self.img_width, self.img_height)
            print('Getting style features from VGG network.')

            self.style_layers = ['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']

            self.style_layer_outputs = []
            for layer in self.style_layers:
                self.style_layer_outputs.append(self.vgg_output_dict[layer])

            style_features = self.get_vgg_style_features(style)
            self.style_features = style_features

            # Style Reconstruction Loss
            if self.style_weight != 0.0:
                for i, layer_name in enumerate(self.style_layers):
                    layer = vgg_layers[layer_name]
                    style_regularizer = StyleReconstructionRegularizer(
                        style_feature_target=style_features[i][0],
                        weight=self.style_weight / len(self.style_layers))
                    style_regularizer.set_layer(layer)
                    layer.regularizers.append(style_regularizer)

            # Feature Reconstruction Loss
            self.content_layer = 'conv3_3'
            self.content_layer_output = self.vgg_output_dict[self.content_layer]

            if self.content_weight != 0.0:
                layer = vgg_layers[self.content_layer]
                content_regularizer = FeatureReconstructionRegularizer(
                    weight=self.content_weight / len(self.content_layer))
                content_regularizer.set_layer(layer)
                layer.regularizers.append(content_regularizer)

        # Total Variation Regularization
        if self.tv_weight != 0.0:
            layer = fastnet_output_layer  # Fastnet Output layer
            tv_regularizer = TVRegularizer(img_width=self.img_width, img_height=self.img_height, weight=self.tv_weight)
            tv_regularizer.set_layer(layer)
            layer.regularizers.append(tv_regularizer)

        if self.model is None:
            self.model = model
        return model

    def _residual_block(self, ip, id):
        init = ip

        x = ReflectionPadding2D()(ip)
        x = Convolution2D(128, self.k, self.k, activation='linear', border_mode='valid',
                          name='res_conv_' + str(id) + '_1')(x)
        x = BatchNormalization(axis=1, mode=self.mode, name="res_batchnorm_" + str(id) + "_1")(x)
        x = Activation('relu', name="res_activation_" + str(id) + "_1")(x)

        x = ReflectionPadding2D()(x)
        x = Convolution2D(self.features, self.k, self.k, activation='linear', border_mode='valid',
                          name='res_conv_' + str(id) + '_2')(x)
        x = BatchNormalization(axis=1, mode=self.mode, name="res_batchnorm_" + str(id) + "_2")(x)

        m = merge([x, init], mode='sum', name="res_merge_" + str(id))
        m = Activation('relu', name="res_activation_" + str(id))(m)

        return m

    def fastnet_predict(self, input_img):
        '''
        Function to get predictions of FastNet model

        Args:
            input_img: input image of shape determined by image_dim_ordering

        Returns: styled (predicted) image

        '''
        if self.fastnet_predict_func is None:
            self.fastnet_predict_func = K.function([self.model.layers[0].input],
                                                   self.fastnet_outputs_dict['fastnet_output'])
        return self.fastnet_predict_func([input_img])

    def get_vgg_style_features(self, input_img):
        '''
        Function to get style features of VGG model

        Args:
            input_img: input image of shape determined by image_dim_ordering

        Returns: list of VGG output features

        '''
        if self.vgg_style_func is None:
            self.vgg_style_func = K.function([self.model.layers[-19].input], self.style_layer_outputs)

        return self.vgg_style_func([input_img])

    def get_vgg_content_features(self, input_img):
        '''
        Function to get content features of VGG model

        Args:
            input_img: input image of shape determined by image_dim_ordering

        Returns: VGG output features

        '''
        if self.vgg_content_func is None:
            self.vgg_content_func = K.function([self.model.layers[-19].input], self.content_layer_output)

        return self.vgg_content_func([input_img])

    def save_fastnet_weights(self, style_name, directory=None):
        '''
        Saves the weights of the FastNet model.

        It creates a temporary save file having the weights of FastNet + VGG,
        loads the weights into just the FastNet model and then deletes the
        FastNet + VGG weights.

        Args:
            style_name: style image name
            directory: base directory of saved weights

        '''
        import os

        if directory is not None:
            if not os.path.exists(directory):
                os.makedirs(directory)

            full_weights_fn = directory + "fastnet_full_model_%s.h5" % style_name
        else:
            full_weights_fn = "fastnet_full_model_%s.h5" % style_name

        self.model.save_weights(filepath=full_weights_fn, overwrite=True)
        f = h5py.File(full_weights_fn)

        layer_names = [name for name in f.attrs['layer_names']]

        fastnet_model = self.create_model()

        for i, layer in enumerate(fastnet_model.layers):
            g = f[layer_names[i]]
            weights = [g[name] for name in g.attrs['weight_names']]
            layer.set_weights(weights)

        if directory is not None:
            weights_fn = directory + "fastnet_%s.h5" % style_name
        else:
            weights_fn = "fastnet_%s.h5" % style_name

        fastnet_model.save_weights(weights_fn, overwrite=True)

        f.close()
        os.remove(full_weights_fn)  # The full weights aren't needed anymore since we only need 1 forward pass
                                    # through the fastnet now.
        print("Saved fastnet weights for style : %s.h5" % style_name)


if __name__ == "__main__":
    from keras.utils.visualize_util import plot

    net = FastStyleNet()
    model = net.create_model(style_image_path=r"D:\Yue\Google Drive\Wallpapers\blue-moon-lake-52859.jpg",
                             train_mode=False)
    model.summary()
    print(len(model.layers))
    plot(model, 'fastnet.png', show_shapes=True, show_layer_names=True)
