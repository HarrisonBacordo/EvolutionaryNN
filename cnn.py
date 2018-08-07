import tensorflow as tf
import numpy as np
import cv2 as cv
from PIL import Image
import glob


def train_network(features):
    tf.Tensor.eval(features)
    input_layer = tf.reshape(features, [-1, 100, 100, 1])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=100,
        kernel_size=[10, 10],
        padding="same",
        activation=tf.nn.sigmoid)

    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=200,
        kernel_size=[10, 10],
        padding="same",
        activation=tf.nn.sigmoid
    )

    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2)

    pool2_flat = tf.reshape(pool2, 25 * 25 * 200)

    dense = tf.layers.dense(inputs=pool2_flat, units=500, activation=tf.nn.sigmoid)

    dropout = tf.layers.dropout(dense, rate=0.6)


def prep_data(data):
    data1 = list()
    for image in data:
        img = tf.image.decode_jpeg(image, channels=1)
        img = tf.image.resize_images(img, [100, 100])
        data1.append(img)

    print(data1)
    return data1


def main():
    image_list = list()
    for filename in glob.glob('dataset/training_set/cats/*.jpg'):
        image_list.append(filename)
    dataset = prep_data(image_list)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for feature in dataset:
            print(sess.run(feature))
            train_network(feature)


if __name__ == '__main__':
    main()
