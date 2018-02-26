import source.hands_bounding_utils.hands_dataset_manager as data
import source.hands_bounding_utils.egohand_dataset_manager as ego
from data_manager import path_manager
import tensorflow as tf
import numpy as np


pm = path_manager.PathManager()


# ############# PATHS ############
train_images_path = data.default_train_images_path()
train_annots_path = data.default_train_annotations_path()
test_images_path = data.default_test_images_path()
test_annots_path = data.default_test_annotations_path()


# ################ BASIC PARAMETERS #####################
train_set_dim = 100
test_set_dim = 100
batch_dimension = 1
learning_rate = 0.005
epochs = 40
height_shrink_rate = 10
width_shrink_rate = 10

# ################## GETTING DATA SETS #######################
train_images, train_heatmaps = data.get_random_samples_from_dataset(train_images_path,
                                                                    train_annots_path, train_set_dim,
                                                                    height_shrink_rate, width_shrink_rate)
test_images, test_heatmaps = data.get_random_samples_from_dataset(test_images_path,
                                                                  test_annots_path, test_set_dim,
                                                                  height_shrink_rate, width_shrink_rate)

# ################## NETWORK PARAMETERS ######################
filters_conv1 = 32
kernel_dim_conv1 = 5

filters_conv2 = 32
kernel_dim_conv2 = 3

filters_conv3 = 32
kernel_dim_conv3 = 3

filters_conv4 = 32
kernel_dim_conv4 = 3

filters_conv5 = 1
kernel_dim_conv5 = 3


pooling_size = [height_shrink_rate, width_shrink_rate]


# ################ NETWORK DEFINITION #########################

# PLACEHOLDERS
x = tf.placeholder(tf.float32, [None, None, None, 3])
y = tf.placeholder(tf.float32, [None, None, None, 1])

# CONVOLUTION 1
conv1 = tf.layers.conv2d(
            inputs=x,
            padding='same',
            filters=filters_conv1,
            kernel_size=[kernel_dim_conv1, kernel_dim_conv1],
            activation=tf.nn.leaky_relu)

# CONVOLUTION 2
conv2 = tf.layers.conv2d(
            inputs=conv1,
            padding='same',
            filters=filters_conv2,
            kernel_size=[kernel_dim_conv2, kernel_dim_conv2],
            activation=tf.nn.leaky_relu)

# CONVOLUTION 3
conv3 = tf.layers.conv2d(
            inputs=conv2,
            padding='same',
            filters=filters_conv3,
            kernel_size=[kernel_dim_conv3, kernel_dim_conv3],
            activation=tf.nn.leaky_relu)

# CONVOLUTION 4
conv4 = tf.layers.conv2d(
            inputs=conv3,
            padding='same',
            filters=filters_conv4,
            kernel_size=[kernel_dim_conv4, kernel_dim_conv4],
            activation=tf.nn.leaky_relu)

# CONVOLUTION 5
conv5 = tf.layers.conv2d(
            inputs=conv4,
            padding='same',
            filters=filters_conv5,
            kernel_size=[kernel_dim_conv5, kernel_dim_conv5],
            activation=tf.nn.leaky_relu)

# POOLING
pool = tf.layers.max_pooling2d(inputs=conv5, pool_size=pooling_size, strides=height_shrink_rate)

# OUTPUT
out = tf.nn.sigmoid(pool)

# ERROR
error = tf.reduce_sum(tf.squared_difference(out, y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(error)

init = tf.global_variables_initializer()

saver = tf.train.Saver()


with tf.Session() as sess:
    # create initialized variables
    sess.run(init)
    # RESTORE PREVIOUSLY TRAINED MODEL, COMMENT IF NO SAVED CHECKPOINTS ARE PRESENT
    # saver.restore(sess, "./model/model.ckpt")
    # TRAINING:
    # -FOR EACH EPOCH
    #   -FOR ECH BATCH
    #       - sess.run trains the network
    total_batch = int(train_set_dim / batch_dimension)
    for epoch in range(epochs):
        tot_cost = 0
        for i in range(0, total_batch):
            batch_x, batch_y = data.get_ordered_batch(train_images, train_heatmaps, batch_dimension, i)
            _, c = sess.run([optimizer, error], feed_dict={x: batch_x, y: batch_y})
            # CALCULATES ERROR AFTER THE EPOCH BY SUMMING ERRORS OF ALL BATCHES
            tot_cost += c
        print("Epoch:", (epoch + 1), "cost =", "{:.5f}".format(tot_cost))

    print("\nTraining complete!")

    # SAVE ACTUAL MODEL, OVERWRITES OLD CHECKPOINT, IF PRESENT
    saver.save(sess, "./model/model.ckpt")
