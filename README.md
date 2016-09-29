# Fast Neural Style in Keras
Implementation of <a href="http://arxiv.org/abs/1603.08155">"Perceptual Losses for Real-Time Style Transfer and Super-Resolution"</a> in Keras 1.1.0.

Keras implementation of <a href="https://github.com/yusuketomoto/chainer-fast-neuralstyle">chainer-fast-neuralstyle by Yusuketomoto</a>.
There are minor differences that are discussed later.

## Examples

Tubingen - Starry Night by Vincent Van Gogh<br>
<img src="https://github.com/titu1994/Fast-Neural-Style/blob/master/images/inputs/tubingen.jpg?raw=true" height=300 width=45%> <img src="https://github.com/titu1994/Fast-Neural-Style/blob/master/images/style/starry_night.jpg?raw=true" height=300 width=45%><br>
<img src="https://github.com/titu1994/Fast-Neural-Style/blob/master/images/generated/tubingen.jpg?raw=true" height=100% width=90%> 

Blue Moon Lake - Starry Night by Vincent Van Gogh<br>
<img src="https://github.com/titu1994/Fast-Neural-Style/blob/master/images/inputs/blue-moon-lake.jpg?raw=true" height=300 width=45%> <img src="https://github.com/titu1994/Fast-Neural-Style/blob/master/images/style/starry_night.jpg?raw=true" height=300 width=45%><br>
<img src="https://github.com/titu1994/Fast-Neural-Style/blob/master/images/generated/blue-moon-lake.jpg?raw=true" height=100% width=90%><br>

## Differences from Original Paper
- The base network does not have the original 3-32-64-128 --- 64-32-3 architecture. Due to certain errors, the model is forced to be
in 3-32-64-64 --- 64-64-3. To compensate, we can use a "wide" mode, which replaces 64 filters with 128.
- The weights of the style loss and content loss are not given in the paper. In this implementation, Content weight has been given 100x 
the weight of style weight (since the style quickky over powers the content).
- A "deep" network can be created, which is useful for larger image_size (say 512)
- Super resolution is not implemented, yet.
- Batch size is 1 instead of 4

## Differences from chainer-fast-neuralstyle
- Model is slightly different than the one used by the chainer implementation.
- Style weight and content weight are different (that uses 10x scale to content)
- Chainer implementation corrects the border effects, which are pronounced in the examples shown here.

# Usage
## Train
The models should be trained on the MS COCO dataset (80k training images). A validation image must be provided to test the intermediate 
stages of the training (since there is no direct validation available, and loss value is not very indicative of performance of the network)

Note that with the default model, each iteration takes 0.2 seconds on a 980M GPU.
```
python train.py "path/to/style/image" "path/to/dataset/" "path/to/validation/image" 
```

There are many parameters that can be changed to facilitate different training behavior. Note that with with the wide and deep model,
each iteration requires roughly 0.65 seconds on a 980M GPU.
```
python train.py "path/to/style/image" "path/to/dataset/" "path/to/validation/image" --content_weight 1e3 
--image_size 512 --model_depth "deep" --model_width "wide" --val_checkpoint 500 --epochs 2
```

A few details to be noted when training:
- At every val_checkpoint (default every 400 samples of MS COCO), the checkpoint weights and validation images will be saved in two folders: val_images and val_weights. 
- At the end of training, another folder with the name "weights" will be created which stores the final weights of the model.
- You can exit at any time using a Keyboard interrupt. This will save the model weights in the "weights" directory.
- You can manually stop the script and rename the validation weights to "fastnet_" + style_name and save them in the weights directory. 

## Prediction
Due to limitations of having to provide output shape for Deconvolution2D layers, it is not possible to transform multiple images using
a single network (unless each image has the same size).

Another limitation is that height and width must be divisible by 4 for the "shallow" model, and divisible by 8 for the "deep" model.
This is because the Deconvolution2D layers need very precise output shape, else they will cause an exception.

```
python transform.py "style name" "path/to/content/image"
```

There are a few parameters which you must use if it is a deep or wide model.
```
python transform.py "style name" "path/to/content/image" --tv_weight 1e-5 --model_depth "deep" --model_width "wide"
```

# Parameters
## Training (train.py)
```
--content_weight: Weight for Content loss. Default = 100
--style_weight: Weight for Style loss. Default = 1
--tv_weight: Weight for Total Variation Regularization. Default = 8.5E-5

--image_size: Image size of training images. Default is 256. Change to 512 for "deep" models
--epochs: Number of epochs to run over training images. Default = 1
--nb_imgs: Number of training images. Default is 80k for the MSCOCO dataset.

--model_depth: Can be one of "shallow" or "deep". Adds more convolution and deconvolution layer for "deep" network. Default = "shallow"
--model_width: Can be one of "thin" or "wide". Changes number of intermediate number of filters. Default = "thin"

--pool_type: Can be one of "max" or "ave". Pooling type to be used. Default = "max"
--kernel_size: Kernel size for convolution and deconvolution layers. Do not change. For testing purposes only.
---val_checkpoint: Iteration count where validation image will be tested. Default is -1, which will produce 200 validation images.
```

## Prediction (transform.py)
```
--tv_weight: Weight for Total Variation Regularization. Default = 8.5E-5

--model_depth: Can be one of "shallow" or "deep". Adds more convolution and deconvolution layer for "deep" network. Default = "shallow"
--model_width: Can be one of "thin" or "wide". Changes number of intermediate number of filters. Default = "thin"

--pool_type: Can be one of "max" or "ave". Pooling type to be used. Default = "max"
--kernel_size: Kernel size for convolution and deconvolution layers. Do not change. For testing purposes only.
```

# Requirements
- Theano
- Keras
- CUDA (GPU) -- Recommended
- CuDNN (GPU) -- Recommended
- Scipy
- Numpy
- PIL / Pillow
- h5py

# Speed
Using the default parameters, this takes 0.2 seconds per iteration on a 980M GPU <br>
Using the "wide" network, this takes 0.3 seconds per iteration on a 980M GPU <br>
Using the "deep" + "wide" network, this takes 0.33 seconds per iteration on a 980M GPU <br>

Using the deep + wide network with image size of 512, this takes 0.65 seconds per iteration on a 980M GPU <br>

# Issues
- Boundry artifacts need to be corrected. This may be due to different tv loss or some other factor.
- keras-rtst trains well using a small number of samples. However this model requires at least 40k iterations to acheive a good result.
- Very string application of style. Content is almost completely destroyed.
