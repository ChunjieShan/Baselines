#!/usr/bin/python3
# -*- coding: utf8 -*-

import tensorflow as tf
import cv2
import sys

model_dir = sys.argv[1]
image_path = sys.argv[2]

model = tf.saved_model.load(model_dir)

image = cv2.imread(image_path)
image = cv2.resize(image, (48, 48))
image = image.reshape(-1, 48, 48, 3)

print(list(model.signatures.keys()))

# infer = model.signatures['serving_default']
# print(infer.structured_outputs)
# # print(prediction)
y = model(image, (-1, 48, 48, 3))
print(y)
