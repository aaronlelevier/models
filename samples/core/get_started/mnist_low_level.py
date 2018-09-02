import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

N_CLASSES = 10
LR = 0.001
EPOCHS = 1
BATCH_SIZE = 32


def load_data():
    mnist = input_data.read_data_sets("MNIST_data/")
    train_x = mnist.train.images
    train_y = np.array(mnist.train.labels,  dtype=np.int32)
    test_x = mnist.test.images
    test_y = np.array(mnist.test.labels, dtype=np.int32)
    return (train_x, train_y), (test_x, test_y)


def get_train_ds(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(1000).repeat().batch(BATCH_SIZE)


def get_test_ds(features, labels):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.batch(BATCH_SIZE)


def conv_net(features):
    net = tf.layers.dense(features, 100, activation='relu')
    return tf.layers.dense(net, N_CLASSES)


def get_train_op(logits, labels):
    loss_op = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=LR)
    return optimizer.minimize(
        loss_op,
        global_step=tf.train.get_global_step())


def main():
    (train_x, train_y), (test_x, test_y) = load_data()

    train_ds = get_train_ds(train_x, train_y)
    train_ds_iterator = train_ds.make_one_shot_iterator()

    test_ds = get_test_ds(test_x, test_y)
    test_ds_iterator = test_ds.make_one_shot_iterator()

    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape=[None, 784], name="X")
    y = tf.placeholder(tf.float32, shape=[None], name="y")

    with tf.Session().as_default() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        # nets
        logits_train = conv_net(X)
        logits_test = conv_net(X)

        # predictions
        pred_classes = tf.argmax(logits_test, axis=1)
        # REVIEW: not sure if used?
        # pred_probas = tf.nn.softmax(pred_classes)

        # train_op
        train_op = get_train_op(
            logits=logits_train,
            labels=tf.cast(y, dtype=tf.int32))

        # accuracy_op - for test set
        accuracy_op = tf.metrics.accuracy(labels=y, predictions=pred_classes)

        for _ in range(EPOCHS):
            # for step in range(0, n_train, BATCH_SIZE):
            X_train_batch_op, y_train_batch_op = train_ds_iterator.get_next()
            X_train_batch, y_train_batch = sess.run([X_train_batch_op, y_train_batch_op])

            train, accuracy = sess.run(
                [train_op, accuracy_op],
                feed_dict={X: X_train_batch, y: y_train_batch})
            import pdb;pdb.set_trace()

        # X_test_batch, y_test_batch = test_ds_iterator.get_next()


if __name__ == '__main__':
    main()
