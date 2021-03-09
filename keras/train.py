import tensorflow as tf

from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.optimizers import Adam
import numpy as np

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

base_model = MobileNetV2(weights='imagenet',
                         include_top=False)  # imports the mobilenet model and discards the last 1000 neuron layer.
x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=preds)

for layer in model.layers:
    layer.trainable = False
# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
    layer.trainable = False
for layer in model.layers[20:]:
    layer.trainable = True

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   brightness_range=(0.0, 0.5),
                                   shear_range=20,
                                   rotation_range=30,
                                   horizontal_flip=True,
                                   vertical_flip=True)  # included in our dependencies

train_generator = train_datagen.flow_from_directory('data/train',
                                                    target_size=(224, 224),
                                                    color_mode='rgb',
                                                    batch_size=32,
                                                    class_mode='categorical',
                                                    shuffle=False)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)  # included in our dependencies

test_generator = train_datagen.flow_from_directory('data/val',
                                                   target_size=(224, 224),
                                                   color_mode='rgb',
                                                   batch_size=32,
                                                   class_mode='categorical',
                                                   shuffle=False)


model.compile(optimizer='Adam', loss=categorical_crossentropy, metrics=['accuracy'])
model.fit(train_generator, batch_size=32, epochs=50, validation_data=test_generator)