from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from process_data import process_data

logDir = '/tmp/my_graph/04/word2vec'

VOCAB_SIZE = 50000
BATCH_SIZE = 128
EMBED_SIZE = 128 # dimension of the word embedding vectors
SKIP_WINDOW = 1 # the context window
NUM_SAMPLED = 64    # Number of negative examples to sample.
LEARNING_RATE = 1.0
NUM_TRAIN_STEPS = 10000
SKIP_STEP = 2000 # how many steps to skip before reporting the loss


def word2vec(batch_gen):
    """ Build the graph for word2vec model and train it """
    # Step 1: define the placeholders for input and output
    # center_words have to be int to work on embedding lookup

    center_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE], name="CenterWord")
    target_words = tf.placeholder(tf.int32, shape=[BATCH_SIZE, SKIP_WINDOW], name="TargetWord")

    # Step 2: define weights. In word2vec, it's actually the weights that we care about
    # vocab size x embed size
    # initialized to random uniform -1 to 1

    init = tf.random_uniform([VOCAB_SIZE, EMBED_SIZE], -1.0, 1.0)
    embed_matrix = tf.Variable(initial_value=init, name="EmbedMatrix")

    # Step 3: define the inference
    # get the embed of input words using tf.nn.embedding_lookup

    embed = tf.nn.embedding_lookup(embed_matrix, center_words, name='embed')

    # Step 4: construct variables for NCE loss
    # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
    # nce_weight (vocab size x embed size), intialized to truncated_normal stddev=1.0 / (EMBED_SIZE ** 0.5)
    # bias: vocab size, initialized to 0

    initW = tf.truncated_normal([VOCAB_SIZE, EMBED_SIZE], mean=0, stddev=1.0/EMBED_SIZE**0.5 )
    nce_weight = tf.Variable(initial_value=initW, name="nce_weights")
    initB = tf.zeros([VOCAB_SIZE])
    nce_bias = tf.Variable(initB, name="nce_bias")

    # define loss function to be NCE loss function
    # tf.nn.nce_loss(weights, biases, labels, inputs, num_sampled, num_classes, ...)
    # need to get the mean accross the batch

    loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weight,
                          biases=nce_bias,
                          labels=target_words,
                          inputs=embed,
                          num_sampled=NUM_SAMPLED,
                          num_classes=VOCAB_SIZE), name="loss")

    # Step 5: define optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    with tf.Session() as sess:
        # initialize variables
        sess.run(tf.global_variables_initializer())

        total_loss = 0.0 # we use this to calculate the average loss in the last SKIP_STEP steps
        writer = tf.summary.FileWriter(logDir, sess.graph)
        for index in range(NUM_TRAIN_STEPS):
            centers, targets= next(batch_gen)
            # create feed_dict, run optimizer, fetch loss_batch
            loss_batch,_ = sess.run([loss, optimizer],
                                    feed_dict={center_words:centers, target_words:targets})

            total_loss += loss_batch
            if (index + 1) % SKIP_STEP == 0:
                print('Average loss at step {}: {:5.1f}'.format(index, total_loss / SKIP_STEP))
                total_loss = 0.0
        writer.close()

def main():
    batch_gen = process_data(VOCAB_SIZE, BATCH_SIZE, SKIP_WINDOW)
    word2vec(batch_gen)

if __name__ == '__main__':
    main()