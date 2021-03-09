import tensorflow as tf
import cv2 as cv
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model = tf.keras.models.load_model("models/fire1.h5")

def fire_detection(image):
    resized = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    resized = cv.resize(resized, (224, 224))

    resized = tf.keras.preprocessing.image.img_to_array(resized)
    resized = tf.keras.applications.mobilenet_v2.preprocess_input(resized)
    resized = np.expand_dims(resized, 0)

    fire, neutral = model.predict([resized])[0]
    print(fire)
    return fire


if __name__ == '__main__':
    image = cv.imread("data/val/Fire/image_0.jpg")
    fire = fire_detection(image)