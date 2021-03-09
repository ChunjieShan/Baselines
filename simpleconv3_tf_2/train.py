#!/usr/bin/python3
# -*- coding: utf8 -*-

import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard
# from dataset import ImageData
from net import simple_conv3_net
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

train_dir = sys.argv[1]
test_dir = sys.argv[2]
batch_size = 10
num_classes = 4
image_size = (96, 96)
learning_rate = 1e-3
epoch = 100
# losses = tf.keras.losses.CategoricalCrossentropy()
losses = sys.argv[3]
optimizer = sys.argv[4]


class TrainingModel:
    def __init__(self,
                 train_dir,
                 test_dir,
                 batch_size,
                 num_classes,
                 image_size,
                 losses,
                 optimizer,
                 epoch=100):

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        self.train_dir = train_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_size = image_size
        self.epoch = epoch
        self.losses = self.get_loss(losses)
        self.optimizer = self.get_optimizer(optimizer)
        self.model = simple_conv3_net(
            (self.image_size[0], self.image_size[1], 3))
        self.compile_model()
        self.train_datagen, self.test_datagen = self.make_datagen()
        self.train_generator, self.test_generator = self.make_flow()
        self.train_num = self.train_generator.samples
        self.test_num = self.test_generator.samples
        history = self.train_model()

    def get_loss(self, loss):
        print("You are using {} loss function!".format(loss))
        if loss == "Categorical":
            return tf.keras.losses.CategoricalCrossentropy()

        elif loss == "Binary":
            return tf.keras.losses.BinaryCrossentropy()

        else:
            raise ValueError(
                "Have you misspelled the name of loss function? Try to check again!"
            )

    def get_optimizer(self, optimizer):
        print("You are using {} optimizer!".format(optimizer))
        if optimizer == "Adam":
            return tf.keras.optimizers.Adam(learning_rate=1e-6)

        elif optimizer == "RMSprop":
            return tf.keras.optimizers.RMSprop(learning_rate=1e-6)

        elif optimizer == "Nesterov":
            return tf.keras.optimizers.SGD(learning_rate=1e-6,
                                           momentum=0.9,
                                           nesterov=True)

        else:
            raise ValueError(
                "Have you misspelled the name of optimizer? Try to check again!"
            )

    def compile_model(self):
        self.model.compile(loss=self.losses,
                           optimizer=self.optimizer,
                           metrics=['acc'])

    def make_datagen(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255,
                                           rotation_range=20,
                                           shear_range=20,
                                           brightness_range=(30, 80))
        # preprocessing_function=self.parse_function)

        test_datagen = ImageDataGenerator(rescale=1. / 255)
        return train_datagen, test_datagen

    def make_flow(self):
        train_generator = self.train_datagen.flow_from_directory(
            self.train_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=True,
            class_mode="categorical")

        test_generator = self.test_datagen.flow_from_directory(
            self.test_dir,
            target_size=self.image_size,
            batch_size=self.batch_size,
            shuffle=False,
            class_mode="categorical")
        return train_generator, test_generator

    def train_model(self):
        log_dir = "./logs"
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        history = self.model.fit(self.train_generator,
                                 steps_per_epoch=self.train_num //
                                 self.batch_size,
                                 workers=1,
                                 epochs=self.epoch,
                                 callbacks=[TensorBoard(log_dir=log_dir)],
                                 validation_data=self.test_generator)

        if not os.path.exists("models"):
            os.mkdir("models")
        self.model.save("./models/model.h5")
        tf.saved_model.save(self.model, "./models/")

        return history

    def parse_function(self, img):
        img = tf.image.random_crop(img,
                                   [self.image_size[0], self.image_size[1], 3])

        return img


if __name__ == "__main__":
    history = TrainingModel(train_dir, test_dir, batch_size, num_classes,
                            image_size, losses, optimizer)
