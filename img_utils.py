from scipy.misc import imread, imresize, imsave, fromimage, toimage
from sklearn.feature_extraction.image import reconstruct_from_patches_2d, extract_patches_2d
from PIL import Image
import numpy as np
import os

from keras import backend as K

aspect_ratio = 0
img_WIDTH = img_HEIGHT = 0, 0


# Util function to open, resize and format pictures into appropriate tensors
def preprocess_image(image_path, img_width=256, img_height=256, load_dims=False, resize=True, size_multiple=4):
    '''
    Preprocess the image so that it can be used by Keras.

    Args:
        image_path: path to the image
        img_width: image width after resizing. Optional: defaults to 256
        img_height: image height after resizing. Optional: defaults to 256
        load_dims: decides if original dimensions of image should be saved,
                   Optional: defaults to False
        vgg_normalize: decides if vgg normalization should be applied to image.
                       Optional: defaults to False
        resize: whether the image should be resided to new size. Optional: defaults to True
        size_multiple: Deconvolution network needs precise input size so as to
                       divide by 4 ("shallow" model) or 8 ("deep" model).

    Returns: an image of shape (3, img_width, img_height) for dim_ordering = "th",
             else an image of shape (img_width, img_height, 3) for dim ordering = "tf"

    '''
    img = imread(image_path, mode="RGB")  # Prevents crashes due to PNG images (ARGB)
    if load_dims:
        global img_WIDTH, img_HEIGHT, aspect_ratio
        img_WIDTH = img.shape[0]
        img_HEIGHT = img.shape[1]
        aspect_ratio = img_HEIGHT / img_WIDTH

    if resize:
        if img_width < 0 or img_height < 0: # We have already loaded image dims
            img_width = (img_WIDTH // size_multiple) * size_multiple # Make sure width is a multiple of 4
            img_height = (img_HEIGHT // size_multiple) * size_multiple # Make sure width is a multiple of 4
        img = imresize(img, (img_width, img_height))

    if K.image_dim_ordering() == "th":
        img = img.transpose((2, 0, 1)).astype(np.float32)
    else:
        img = img.astype(np.float32)

    img = np.expand_dims(img, axis=0)
    return img


# Util function to convert a tensor into a valid image
def deprocess_image(x):
    '''
    Removes the pre processing steps applied to image.

    Args:
        x: input image of shape (3, img_width, img_height) [th],
           or input image of shape (img_width, img_height, 3) [tf]
        denormalize_vgg: whether vgg normalization should be reversed

    Returns: image of same shape as input shape

    '''
    if K.image_dim_ordering() == "th":
        x = x.transpose((1, 2, 0))

    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Util function to preserve image color
def original_color_transform(content, generated):
    '''
    Applies the color space of content image to the generated image

    Args:
        content: input image of shape (img_width, img_height, 3)
        generated: input image of shape (img_width, img_height, 3)

    Returns: image of same shape as input shape

    '''
    generated = fromimage(toimage(generated), mode='YCbCr')  # Convert to YCbCr color space
    generated[:, :, 1:] = content[:, :, 1:]  # Generated CbCr = Content CbCr
    generated = fromimage(toimage(generated, mode='YCbCr'), mode='RGB')  # Convert to RGB color space
    return generated


# Util function to save intermediate images
def save_result(img, fname, img_width=None, img_height=None, preserve_color=False, content_img_path=None, directory=None):
    '''
    Save the resultant image

    Args:
        img: input image of shape (img_width, img_height, 3)
        fname: filename of output image
        img_width: resize dimension
        img_height: resize dimension
        preserve_color: whether to preserve original color of the content image
        content_img_path: path to content image. Optional, but required if color preservation is required
        directory: base directory where image will be stored

    '''
    if directory is not None:
        if not os.path.exists(directory):
            os.makedirs(directory)

        fname = directory + fname

    # We require original image if we are to preserve color in YCbCr mode
    if preserve_color:
        assert content_img_path is not None, \
            "If color is to be preserved, then content image path must be given as well."

        content = imread(content_img_path, mode="YCbCr")
        if img_width is not None and img_height is not None:
            content = imresize(content, (img_width, img_height))
        img = original_color_transform(content, img)

    imsave(fname, img)

def _check_image(path, i, nb_images):
    '''
    Test if image can be loaded by PIL. If image cannot be loaded, delete it from dataset.

    Args:
        path: path to image
        i: iteration number
        nb_images: total number of images

    '''
    try:
        im = Image.open(path)
        im.verify()
        im = Image.open(path)
        im.load()
        if i % 1000 == 0: print('%0.2f percent images are checked.' % (i * 100 / nb_images))
    except:
        os.remove(path)
        print("Image number %d is corrupt (path = %s). Deleting from dataset." % (i, path))

def check_dataset(path):
    '''
    Use to check the dataset for any jpg corruptions.
    If there exists corruption in an image, then it is deleted.

    Note:
        If due to some reason the corrupted file cannot be deleted via os.remove(),
        it will print the path of the file. Please delete the file manually in such a case.

    Args:
        path: Path to the dataset of images
    '''
    from multiprocessing.pool import Pool
    pool = Pool()

    nb_images = len([name for name in os.listdir(path)])
    print("Checking %d images" % nb_images)

    for i, file in enumerate(os.listdir(path)):
        pool.apply_async(_check_image, args=(path + "\\" + file, i, nb_images))

        if i % 1000 == 0: print('%0.2f percent images are added to queue.' % (i * 100 / nb_images))

    pool.close()
    pool.join()

    new_nb_images = len([name for name in os.listdir(path)])
    print()
    print("New size of dataset : %d. Number of images deleted = %d" % (new_nb_images, nb_images - new_nb_images))

if __name__ == "__main__":
    '''
    Run this script to check for corrupt images in an image dataset whose path is provided
    '''
    ms_coco_path = r""

    '''
    Note:
        If due to some reason the corrupted file cannot be deleted via os.remove(),
        it will print the path of the file. Please delete the file manually in such a case.
    '''
    check_dataset(ms_coco_path)


