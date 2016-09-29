import models
from loss import dummy_loss
import img_utils
import time
import os
import argparse

parser = argparse.ArgumentParser(description='Fast Neural style transfer with Keras.')
parser.add_argument("style_name", type=str, help='Exact name of the style (without the fastnet_)')
parser.add_argument("content_img", type=str, help='Content image path')

parser.add_argument("--tv_weight", type=float, default=8.5e-5, help='Total Variation Weight')

parser.add_argument("--model_depth", type=str, default='shallow', help='Can be one of "shallow" or "deep"')
parser.add_argument("--model_width", default="thin", type=str, help='Can be one of "thin" or "wide"')

parser.add_argument("--pool_type", default="max", type=str, help='Pooling type')
parser.add_argument("--kernel_size", default=3, type=int, help='Kernel Size')

args = parser.parse_args()

''' Attributes '''
style_name = str(args.style_name)
content_path = str(args.content_img)

tv_weight = float(args.tv_weight)

pool_type = str(args.pool_type)
assert pool_type in ["max", "ave"], 'Pool Type must be either "max" or "ave"'
pool_type = 1 if pool_type == "ave" else 0

kernel_size = int(args.kernel_size)

model_depth = str(args.model_depth).lower()
assert model_depth in ["shallow", "wide"], 'model_depth must be one of "shallow" or "wide"'

deep_model = False if args.model_depth == "shallow" else True
size_multiple = 8 if deep_model else 4

model_width = str(args.model_width).lower()
assert model_width in ["thin", "wide"], 'model_width must be one of "thin" or "wide"'

content_name = os.path.splitext(os.path.basename(content_path))[0]
output_image = "%s_%s.png" % (content_name, style_name)

''' Transform image '''

img = img_utils.preprocess_image(content_path, load_dims=True, resize=True, img_width=-1,
                                 img_height=-1, size_multiple=size_multiple)
img /= 255.
width, height = img.shape[2], img.shape[3]

FastNet = models.FastStyleNet(img_width=width, img_height=height, kernel_size=kernel_size, pool_type=pool_type,
                              tv_weight=tv_weight, model_width=model_width, model_depth=model_depth)
FastNet.create_model(style_name=style_name)
FastNet.model.compile(optimizer="adadelta", loss=dummy_loss)

t1 = time.time()
output = FastNet.fastnet_predict(img)
t2 = time.time()

print("Prediction time : %0.2f seconds" % (t2 - t1))

img = output[0, :, :, :]
img = img_utils.deprocess_image(img)

img_utils.save_result(img, output_image, width, height)
