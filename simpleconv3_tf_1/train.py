#!/usr/bin/python3
# -*- coding: utf8 -*-

from dataset import *
from net import simple_conv3_net
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import sys
import numpy as np
import tensorflow as tf
import os
import cv2

txt_file = sys.argv[1]
batch_size = 64
num_classes = 2
image_size = (48, 48)
learning_rate = 0.001

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


class TrainingModel():
    '''
    It's a image classification baseline which was built by TensorFlow 1.13.
    '''
    def __init__(self, debug=True):
        '''
        Initializing some params and loading data.
        '''
        self.debug = debug
        self.dataset = ImageData(txt_file, batch_size, num_classes, image_size)
        self.iterator = self.dataset.data.make_one_shot_iterator()
        self.dataset_size = self.dataset.dataset_size
        self.batch_images, self.batch_labels = self.iterator.get_next()
        self.y_logits = simple_conv3_net(self.batch_images)

    def train_model(self):
        '''
        Training the model by building graph.
        '''
        y = self.y_logits
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits( # define loss function
            logits=self.y_logits,
            labels=self.batch_labels)
        cross_entropy = tf.reduce_mean(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(y, 1),
                                      tf.argmax(self.batch_labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,
                                          tf.float32)) # define acurracy

        update_ops = tf.get_collection(
            tf.GraphKeys.UPDATE_OPS) # get the update operation
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(learning_rate).minimize(
                cross_entropy) # params optimization

        saver = tf.train.Saver()
        in_steps = 100
        checkpoint_dir = "checkpoints/"
        if not os.path.exists(checkpoint_dir):
            os.mkdir(checkpoint_dir)
        log_dir = 'logs/'
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

        summary = tf.summary.FileWriter(logdir=log_dir)
        loss_summary = tf.summary.scalar("loss", cross_entropy)
        acc_summary = tf.summary.scalar("acc", accuracy)
        image_summary = tf.summary.image("image", self.batch_images)

        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            steps = 10000
            for i in range(steps):
                _, cross_entropy_, accuracy_, batch_images_, batch_labels_, loss_summary_, acc_summary_, image_summary_ = sess.run(
                    [
                        train_step, cross_entropy, accuracy, self.batch_images,
                        self.batch_labels, loss_summary, acc_summary,
                        image_summary
                    ])
                if i % in_steps == 0:
                    print("Iteration: {}, Loss: {:.4f}, Acc: {:.4%}".format(
                        i, cross_entropy_, accuracy_))

                    saver.save(sess,
                               checkpoint_dir + 'model.ckpt',
                               global_step=i)
                    summary.add_summary(loss_summary_, i)
                    summary.add_summary(acc_summary_, i)
                    summary.add_summary(image_summary_, i)
                    print("Predict: ", self.y_logits, "Labels: ",
                          self.batch_labels)

                    if self.debug:
                        image_debug = batch_images_[0].copy()
                        image_debug = np.squeeze(image_debug)
                        print(image_debug, image_debug.shape)
                        print(np.max(image_debug))
                        image_label = batch_labels_[0].copy()
                        print(np.squeeze(image_label))

                        image_debug = cv2.cvtColor(
                            (image_debug * 255).astype(np.uint8),
                            cv2.COLOR_RGB2BGR)
                        cv2.namedWindow("Debug Image", 0)
                        cv2.imshow("Debug Image", image_debug)
                        k = cv2.waitKey(0)
                        if k == ord('q'):
                            break


if __name__ == "__main__":
    history = TrainingModel(debug=False)
    history.train_model()
