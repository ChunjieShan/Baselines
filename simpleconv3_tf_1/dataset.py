#!/usr/bin/python3
# -*- coding: utf8 -*-

import tensorflow as tf
import sys
from tensorflow.data import Dataset
from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor


class ImageData:
    def read_txt_file(self):
        self.img_paths = []
        self.labels = []
        for line in open(self.txt_file, 'r'):
            items = line.split(' ')
            self.img_paths.append(items[0])
            new_label = items[1].split('\n')
            self.labels.append(int(new_label[0]))

    def __init__(self,
                 txt_file,
                 batch_size,
                 num_classes,
                 image_size,
                 buffer_scale=100):
        self.txt_file = txt_file
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_size = image_size
        buffer_size = buffer_scale * batch_size

        self.read_txt_file()
        self.dataset_size = len(self.labels)
        print("The dataset has {} data.".format(self.dataset_size))

        # Converting images and labels to tensor
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        self.labels = convert_to_tensor(self.labels, dtype=dtypes.int32)

        # Creating dataset
        data = Dataset.from_tensor_slices((self.img_paths, self.labels))
        data = data.map(self.parse_function)
        data = data.repeat(1000)
        data = data.shuffle(buffer_size=buffer_size)

        # Setting data batch
        self.data = data.batch(batch_size)

    def parse_function(self, file_name, label):
        label_ = tf.one_hot(label, self.num_classes)
        img = tf.read_file(file_name)
        img = tf.image.decode_jpeg(img)
        img = tf.image.convert_image_dtype(img, dtype=tf.float32)
        img = tf.random_crop(img, [self.image_size[0], self.image_size[1], 3])
        img = tf.image.random_flip_left_right(img)
        img = self.augment_dataset(img, self.image_size)

        return img, label_

    def augment_dataset(self, img, image_size):
        img = tf.image.random_brightness(img, max_delta=63)
        img = tf.image.random_contrast(img, lower=0.2, upper=1.8)
        img = tf.image.per_image_standardization(img)

        return img


if __name__ == "__main__":
    txt_file = sys.argv[1]
    ImageData(txt_file, 3, 2, (48, 48))
