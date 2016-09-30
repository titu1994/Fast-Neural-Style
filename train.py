from __future__ import print_function
from __future__ import division

import os
from loss import dummy_loss
import models
import numpy as np
import argparse
import time
import img_utils

from keras.optimizers import Adadelta
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

# Only supports Theano for now
K.set_image_dim_ordering("th")

parser = argparse.ArgumentParser(description='Fast Neural style transfer with Keras.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')

parser.add_argument("data_path", type=str, help="Path to training images")
parser.add_argument("validation_img", type=str, default=None, help='Path to validation image')

parser.add_argument("--content_weight", type=float, default=10., help='Content weight')
parser.add_argument("--style_weight", type=float, default=1., help='Style weight')
parser.add_argument("--tv_weight", type=float, default=8.5e-5, help='Total Variation Weight')

parser.add_argument("--image_size", dest="img_size", default=256, type=int, help='Output Image size')
parser.add_argument("--epochs", default=1, type=int, help='Number of epochs')
parser.add_argument("--nb_imgs", default=80000, type=int, help='Number of images per epoch')

parser.add_argument("--model_depth", default="shallow", type=str, help='Can be one of "shallow" or "wide"')
parser.add_argument("--model_width", default="thin", type=str, help='Can be one of "thin" or "wide"')

parser.add_argument("--pool_type", default="max", type=str, help='Pooling type')
parser.add_argument("--kernel_size", default=3, type=int, help='Kernel Size')
parser.add_argument("--val_checkpoint", type=int, default=-1, help='Check the output of network to a validation image')


args = parser.parse_args()
style_reference_image_path = args.style_reference_image_path
style_name = os.path.splitext(os.path.basename(style_reference_image_path))[0]

validation_img_path = args.validation_img

''' Attributes '''
# Dimensions of the input image
img_width = img_height = int(args.img_size)  # Image size needs to be same for gram matrix
nb_epoch = int(args.epochs)
num_iter = int(args.nb_imgs)  # Should be equal to number of images
train_batchsize = 1  # Using batchsize >= 2 results in unstable training

content_weight = float(args.content_weight)
style_weight = float(args.style_weight)
tv_weight = float(args.tv_weight)

val_checkpoint = int(args.val_checkpoint)
if val_checkpoint == -1:
    val_checkpoint = num_iter / 200 # Assuming full MS COCO Dataset has ~80k samples, validate every 400 samples

kernel_size = int(args.kernel_size)

pool_type = str(args.pool_type)
assert pool_type in ["max", "ave"], 'Pool Type must be either "max" or "ave"'
pool_type = 1 if pool_type == "ave" else 0

iteration = 0

model_depth = str(args.model_depth).lower()
assert model_depth in ["shallow", "deep"], 'model_depth must be one of "shallow" or "deep"'

model_width = str(args.model_width).lower()
assert model_width in ["thin", "wide"], 'model_width must be one of "thin" or "wide"'

''' Model '''

if not os.path.exists("models/"):
    os.makedirs("models/")

FastNet = models.FastStyleNet(img_width=img_width, img_height=img_height, kernel_size=kernel_size, pool_type=pool_type,
                              style_weight=style_weight, content_weight=content_weight, tv_weight=tv_weight,
                              model_width=model_width, model_depth=model_depth,
                              save_fastnet_model="models/%s.h5" % style_name)

model = FastNet.create_model(style_name=None, train_mode=True, style_image_path=style_reference_image_path)

optimizer = Adadelta()
model.compile(optimizer, dummy_loss)  # Dummy loss is used since we are learning from regularizes
print('Finished compiling fastnet model.')

datagen = ImageDataGenerator(rescale=1. / 255)

if K.image_dim_ordering() == "th":
    dummy_y = np.zeros((train_batchsize, 3, img_height, img_width))  # Dummy output, not used since we use regularizers to train
else:
    dummy_y = np.zeros((train_batchsize, img_height, img_width, 3)) # Dummy output, not used since we use regularizers to train

prev_improvement = -1
early_stop = False
validation_fastnet = None

for i in range(nb_epoch):
    print()
    print("Epoch : %d" % (i + 1))

    for x in datagen.flow_from_directory(args.data_path, class_mode=None, batch_size=train_batchsize,
                                         target_size=(img_width, img_height), shuffle=False):
        try:
            t1 = time.time()
            hist = model.fit([x, x.copy()], dummy_y, batch_size=train_batchsize, nb_epoch=1, verbose=0)

            iteration += train_batchsize
            loss = hist.history['loss'][0]

            if prev_improvement == -1:
                prev_improvement = loss

            improvement = (prev_improvement - loss) / prev_improvement * 100
            prev_improvement = loss

            t2 = time.time()
            print("Iter : %d / %d | Improvement : %0.2f percent | Time required : %0.2f seconds | Loss : %d" %
                 (iteration, num_iter, improvement, t2 - t1, loss))

            if iteration % val_checkpoint == 0:
                print("Producing validation image...")
                x = img_utils.preprocess_image(validation_img_path, resize=False)
                x /= 255.

                width, height = x.shape[2], x.shape[3]

                iter_path = style_name + "_epoch_%d_at_iteration_%d" % (i + 1, iteration)
                FastNet.save_fastnet_weights(iter_path, directory="val_weights/")

                path = "val_weights/fastnet_" + iter_path + ".h5"

                if validation_fastnet is None:
                    validation_fastnet = models.FastStyleNet(width, height, kernel_size, pool_type,
                                                             model_width=model_width, model_depth=model_depth)
                    validation_fastnet.create_model(validation_path=path)
                    validation_fastnet.model.compile(optimizer, dummy_loss)
                else:
                    validation_fastnet.model.load_weights(path)

                y_hat = validation_fastnet.fastnet_predict(x)

                y_hat = y_hat[0, :, :, :]
                y_hat = y_hat.transpose((1, 2, 0))
                y_hat = np.clip(y_hat, 0, 255).astype('uint8')

                path = "val_epoch_%d_at_iteration_%d.png" % (i + 1, iteration)
                img_utils.save_result(y_hat, path, directory="val_imgs/")

                path = "val_imgs/" + path
                print("Validation image saved at : %s" % path)

            if iteration >= num_iter:
                break

        except KeyboardInterrupt:
            print("Keyboard interrupt detected. Stopping early.")
            early_stop = True
            break

    iteration = 0

    if early_stop:
        break

FastNet.save_fastnet_weights(style_name, directory="weights/")
