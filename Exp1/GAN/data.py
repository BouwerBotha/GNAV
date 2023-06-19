import os
import random

import tensorflow as tf
from PIL import Image
import pathlib
from common import WIDTH,HEIGHT,NUM_DIM
IMG_PATH = 'images'
img_dir = pathlib.Path(IMG_PATH)


BATCH_SIZE = 16
val_size = 100
AUTOTUNE = tf.data.AUTOTUNE


def process_img(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_png(img, channels=NUM_DIM)
    img = tf.image.resize(img, [HEIGHT, WIDTH])
    img = scale_image(img)
    return img


def get_data():
    fake_ds = tf.data.Dataset.list_files(str(img_dir / 'fake/*'))  ##returns dataset containing list of file strings
    real_ds = tf.data.Dataset.list_files(str(img_dir / 'real/*'))

    real_ds = real_ds.map(process_img)  ##Convert file strings to images
    fake_ds = fake_ds.map(process_img)
    real_ds = real_ds.shuffle(100)
    fake_ds = fake_ds.shuffle(100)
    real_ds = optimizeds(real_ds)
    fake_ds = optimizeds(fake_ds)
    return fake_ds, real_ds


def optimizeds(ds):
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

#converts normalized float image back to byte array
def reverse_parse(images):
    return tf.cast(tf.maximum(0.0, tf.minimum(255.0, ((images + 1) / 2) * 255.0)), tf.uint8)
#used to scale array values to 0.0-1.0
def scale_image(img):
    return img/127.5 - 1.0


def duplicate_real():
    k = 3000
    for file in os.listdir('images/real'):
        name = 'images/real/'+file
        im = Image.open(name)
        im_flipped = tf.image.flip_left_right(im)
        im_flipped = tf.image.resize(im_flipped,[54,72],method=tf.image.ResizeMethod.BICUBIC)
        im_flipped = tf.image.random_crop(im_flipped,[HEIGHT,WIDTH,NUM_DIM])
        tf.keras.utils.save_img('images/real/{0}.png'.format(k),im_flipped,scale=False)
        k+=1

