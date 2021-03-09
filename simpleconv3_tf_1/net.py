import tensorflow as tf


def simple_conv3_net(x, debug=True):
    with tf.variable_scope("allconv4"):
        conv1 = tf.layers.conv2d(
            x,
            name="conv1",
            filters=12,
            kernel_size=3,
            strides=2,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())

        bn1 = tf.layers.batch_normalization(conv1, name="bn1", training=True)

        conv2 = tf.layers.conv2d(
            bn1,
            name="conv2",
            filters=24,
            kernel_size=3,
            strides=2,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())

        bn2 = tf.layers.batch_normalization(conv2, name='bn2', training=True)

        conv3 = tf.layers.conv2d(
            bn2,
            name="conv3",
            filters=48,
            kernel_size=3,
            strides=2,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer(),
            bias_initializer=tf.contrib.layers.xavier_initializer())

        bn3 = tf.layers.batch_normalization(conv3, name='bn3', training=True)

        conv3_flat = tf.reshape(bn3, [-1, 5 * 5 * 48])

        dense = tf.layers.dense(
            conv3_flat,
            name="dense",
            units=128,
            activation=tf.nn.relu,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        logits = tf.layers.dense(
            dense,
            name="logits",
            units=2,
            activation=tf.nn.softmax,
            kernel_initializer=tf.contrib.layers.xavier_initializer())

        if debug:
            print("X Shape: ", x.shape)
            print("Conv1 Shape: ", conv1.shape)
            print("bn1 Shape: ", bn1.shape)
            print("Conv2 Shape: ", conv2.shape)
            print("bn2 Shape: ", bn2.shape)
            print("Conv3 Shape: ", conv3.shape)
            print("bn3 Shape: ", bn3.shape)

        return logits


if __name__ == "__main__":
    import numpy as np
    x = np.random.randn(1, 48, 48, 3)
    print(x.shape)
    model = simple_conv3_net(tf.convert_to_tensor(x, dtype=tf.float32),
                             debug=True)
