""" Using convolutional net on MNIST dataset of handwritten digit
(http://yann.lecun.com/exdb/mnist/)
"""
from __future__ import print_function

import os
import time 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.summary.writer import writer

N_CLASSES = 10
N_ROWS = 28
N_COLS = 28
KERNEL_SIZE = 5

N_CHANNEL_1 = 32
N_CHANNEL_2 = 64

N_CHANNEL_FC1 = 1024

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets("/tmp/data/mnist", one_hot=True)

# Step 2: Define paramaters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 50

logDir = '/tmp/my_graph/07/convnet'
## Remove the files in the log dir
'''
if tf.gfile.Exists(logDir):
    tf.gfile.DeleteRecursively(logDir)
tf.gfile.MakeDirs(logDir)
tf.gfile.MakeDirs(logDir + '/checkpoints')
'''

class MNIST_CNN:
    def __init__(self, learningRate, nClasses):
        self.learningRate = learningRate
        self.nClasses = nClasses

        self.nRows = N_ROWS
        self.nCols = N_COLS
        self.kernelSize = KERNEL_SIZE
        self.nChannel1 = N_CHANNEL_1
        self.nChannel2 = N_CHANNEL_2
        self.nChannelFC = N_CHANNEL_FC1
        self.nClasses = N_CLASSES

        # Variable that contain the number of training step.
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                       name='global_step')

    def _create_placeholders(self):
        # Step 3: create placeholders for features and labels
        # each image in the MNIST data is of shape 28*28 = 784
        # therefore, each image is represented with a 1x784 tensor
        # We'll be doing dropout for hidden layer so we'll need a placeholder
        # for the dropout probability too
        # Use None for shape so we can change the batch_size once we've built the graph
        with tf.name_scope('data'):
            self.X = tf.placeholder(tf.float32, [None, self.nRows*self.nCols], name="X_placeholder")
            self.Y = tf.placeholder(tf.float32, [None, self.nClasses], name="Y_placeholder")

        self.dropout = tf.placeholder(tf.float32, name='dropout')

    def _create_conv_layer1(self):
        with tf.variable_scope('conv1') as scope:
            # first, reshape the image to [BATCH_SIZE, 28, 28, 1] to make it work with tf.nn.conv2d
            # use the dynamic dimension -1

            images = tf.reshape(self.X, shape=[-1, self.nRows, self.nCols, 1], )

            # create kernel variable of dimension [5, 5, 1, 32]
            # use tf.truncated_normal_initializer()
            kernel = tf.get_variable('kernel',
                                     shape=[self.kernelSize, self.kernelSize, 1, self.nChannel1],
                                     initializer=tf.truncated_normal_initializer())

            # create biases variable of dimension [32]
            # use tf.random_normal_initializer()
            bias = tf.get_variable('bias', shape=[self.nChannel1],
                                   initializer=tf.random_normal_initializer())

            # apply tf.nn.conv2d. strides [1, 1, 1, 1], padding is 'SAME'
            conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')

            # apply relu on the sum of convolution output and biases
            self.conv1 = tf.nn.relu(conv + bias, name=scope.name)

            # output is of dimension BATCH_SIZE x 28 x 28 x 32

    def _create_pool_layer1(self):
        with tf.variable_scope('pool1') as scope:
            # apply max pool with ksize [1, 2, 2, 1], and strides [1, 2, 2, 1], padding 'SAME'

            self.pool1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1],
                                        strides=[1, 2, 2, 1], padding="SAME", name=scope.name)

            # output is of dimension BATCH_SIZE x 14 x 14 x 32

    def _create_conv_layer2(self):
        with tf.variable_scope('conv2') as scope:
            # similar to conv1, except kernel now is of the size 5 x 5 x 32 x 64
            kernel = tf.get_variable('kernels', [KERNEL_SIZE, KERNEL_SIZE, N_CHANNEL_1, N_CHANNEL_2],
                                initializer=tf.truncated_normal_initializer())
            biases = tf.get_variable('biases', [N_CHANNEL_2],
                                initializer=tf.random_normal_initializer())
            conv = tf.nn.conv2d(self.pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
            self.conv2 = tf.nn.relu(conv + biases, name=scope.name)

            # output is of dimension BATCH_SIZE x 14 x 14 x 64

    def _create_pool_layer2(self):
        with tf.variable_scope('pool2') as scope:
            # similar to pool1
            self.pool2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                                    padding='SAME')

            # output is of dimension BATCH_SIZE x 7 x 7 x 64

    def _create_fc_layer(self):
        with tf.variable_scope('fc') as scope:
            # use weight of dimension 7 * 7 * 64 x 1024
            input_features = 7 * 7 * 64

            # create weights and biases
            W = tf.get_variable('weights', shape= [input_features, N_CHANNEL_FC1],
                                      initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [N_CHANNEL_FC1],
                                initializer=tf.random_normal_initializer())

            # reshape pool2 to 2 dimensional
            pool2 = tf.reshape(self.pool2, [-1, input_features])

            # apply relu on matmul of pool2 and w + b

            input = tf.add(tf.matmul(pool2, W), b)
            fc = tf.nn.relu(input, name='relu')

            # apply dropout
            self.fc = tf.nn.dropout(fc, self.dropout, name='relu_dropout')

    def _create_softmax_layer(self):
        with tf.variable_scope('softmax_linear') as scope:
            # this you should know. get logits without softmax
            # you need to create weights and biases
            W = tf.get_variable('weights', shape= [N_CHANNEL_FC1, N_CLASSES],
                                      initializer=tf.truncated_normal_initializer())
            b = tf.get_variable('biases', [N_CLASSES],
                                initializer=tf.random_normal_initializer())
            self.logits = tf.add(tf.matmul(self.fc, W), b)

    def _create_loss(self):
        # Step 6: define loss function
        # use softmax cross entropy with logits as the loss function
        # compute mean cross entropy, softmax is applied internally
        with tf.name_scope('loss'):
            # you should know how to do this too
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def _create_optimizer(self):
        # Step 7: define training op
        # using gradient descent with learning rate of LEARNING_RATE to minimize cost
        # don't forgot to pass in global_step
        # The Global Step shall be automatically updated
        self.optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(self.loss,
                                                                                   global_step=self.global_step)
    def _create_summaries(self):
        with tf.name_scope("summaries"):
            tf.summary.scalar("loss", self.loss)
            tf.summary.histogram("histogram loss", self.loss)
            # because you have several summaries, we should merge them all
            # into one op to make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        """ Build the graph for our model """
        self._create_placeholders()

        self._create_conv_layer1()
        self._create_pool_layer1()

        self._create_conv_layer2()
        self._create_pool_layer2()

        self._create_fc_layer()
        self._create_softmax_layer()

        self._create_loss()
        self._create_optimizer()
        self._create_summaries()

#############################################################################


def train_model(model, num_epochs, batch_size, dropout, skip_step):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        # to visualize using TensorBoard
        writer_training = tf.summary.FileWriter(logDir + '/train', sess.graph)
        writer_test = tf.summary.FileWriter(logDir + '/test', sess.graph)

        ##### You have to create folders to store checkpoints
        ckpt = tf.train.get_checkpoint_state(
            os.path.dirname(logDir + '/checkpoints/checkpoint'))

        # if that checkpoint exists, restore from checkpoint
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        initial_step = model.global_step.eval()

        start_time = time.time()
        n_batches = int(mnist.train.num_examples / batch_size)

        total_loss = 0.0
        for index in range(initial_step, n_batches * num_epochs): # train the model n_epochs times
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            feed_dict = {model.X: X_batch, model.Y: Y_batch, model.dropout: dropout}
            _, loss_batch, summary = sess.run([model.optimizer, model.loss, model.summary_op],
                                              feed_dict=feed_dict)

            writer_training.add_summary(summary=summary, global_step=index)

            total_loss += loss_batch
            if (index + 1) % skip_step == 0:
                print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / skip_step))
                total_loss = 0.0
                saver.save(sess, logDir + '/checkpoints/mnist-convnet', index)


        print("Optimization Finished!") # should be around 0.35 after 25 epochs
        print("Total time: {0} seconds".format(time.time() - start_time))

        # test the model
        n_batches = int(mnist.test.num_examples/batch_size)
        total_correct_preds = 0
        for i in range(n_batches):
            X_batch, Y_batch = mnist.test.next_batch(batch_size)
            feed_dict = {model.X: X_batch, model.Y: Y_batch, model.dropout: dropout}
            _, loss_batch, logits_batch, summary_test = sess.run([model.optimizer, model.loss,
                                                             model.logits, model.summary_op],
                                                            feed_dict=feed_dict)

            writer_test.add_summary(summary=summary_test, global_step=i)

            preds = tf.nn.softmax(logits_batch)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
            accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
            total_correct_preds += sess.run(accuracy)

        print("Accuracy {0}".format(total_correct_preds/mnist.test.num_examples))
        writer_training.close()
        writer_test.close()

def main():
    model = MNIST_CNN(LEARNING_RATE, N_CLASSES)
    model.build_graph()
    train_model(model, N_EPOCHS, BATCH_SIZE, DROPOUT, SKIP_STEP)

if __name__ == '__main__':
    main()