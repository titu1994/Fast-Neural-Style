from loss import dummy_loss
import img_utils
import layers
import time
import os
import argparse

from keras.models import load_model

parser = argparse.ArgumentParser(description='Fast Neural style transfer with Keras.')
parser.add_argument("style_name", type=str, help='Exact name of the style (without the fastnet_)')
parser.add_argument("content_img", type=str, help='Content image path')

parser.add_argument("--tv_weight", type=float, default=8.5e-5, help='Total Variation Weight')

args = parser.parse_args()

''' Attributes '''
style_name = str(args.style_name)
content_path = str(args.content_img)

tv_weight = float(args.tv_weight)

content_name = os.path.splitext(os.path.basename(content_path))[0]
output_image = "%s_%s.png" % (content_name, style_name)

''' Transform image '''

model_path = "models/" + style_name + ".h5"
weights_path = "weights/fastnet_%s.h5" % style_name

with open(model_path, "r") as f:
    string = f.read()
    model = load_model(
        model_path,
        dict(Denormalize=layers.Denormalize, VGGNormalize=layers.VGGNormalize))
    model.compile("adadelta", dummy_loss)

    model.load_weights(weights_path)

size_multiple = 4 if len(model.layers) == 58 else 8 # 58 layers in shallow model, 62 in deeper model

img = img_utils.preprocess_image(content_path, load_dims=True, resize=True, img_width=-1,
                                 img_height=-1, size_multiple=size_multiple)
img /= 255.
width, height = img.shape[2], img.shape[3]

t1 = time.time()
output = model.predict_on_batch(img)
t2 = time.time()

print("Saved image : %s" % output_image)
print("Prediction time : %0.2f seconds" % (t2 - t1))

img = output[0, :, :, :]
img = img_utils.deprocess_image(img)

img_utils.save_result(img, output_image, width, height)
