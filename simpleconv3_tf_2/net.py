#!/usr/bin/python3
# -*- coding: utf8 -*-

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, Flatten, Dense

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def simple_conv3_net(input_shape):
    input = tf.keras.layers.Input(input_shape)
    x = Conv2D(32, (3, 3), 2, activation=tf.nn.relu)(input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), 2, activation=tf.nn.relu)(x)
    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), 2, activation=tf.nn.relu)(x)
    # x = Flatten()(x)
    x = tf.keras.layers.GlobalAvgPool2D()(x)
    output = Dense(4, activation=tf.nn.softmax)(x)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model


if __name__ == "__main__":
    import numpy as np
    img = np.random.randn(1, 160, 160, 3)
    model = simple_conv3_net(input_shape=(190, 190, 3))
    prediction = model(img)
    print(prediction)
    # print(output)
