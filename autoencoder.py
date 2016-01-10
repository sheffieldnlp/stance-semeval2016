""" Deep Auto-Encoder implementation

    An auto-encoder works as follows:

    Data of dimension k is reduced to a lower dimension j using a matrix multiplication:
    softmax(W*x + b)  = x'

    where W is matrix from R^k --> R^j

    A reconstruction matrix W' maps back from R^j --> R^k

    so our reconstruction function is softmax'(W' * x' + b')

    Now the point of the auto-encoder is to create a reduction matrix (values for W, b)
    that is "good" at reconstructing  the original data.

    Thus we want to minimize  ||softmax'(W' * (softmax(W *x+ b)) + b')  - x||

    A deep auto-encoder is nothing more than stacking successive layers of these reductions.
"""
import tensorflow as tf
import numpy as np
import math
import random
import tokenize_tweets
from tokenize_tweets import convertTweetsToVec, readTweetsOfficial


def create(x, layer_sizes):
    # Build the encoding layers
    next_layer_input = x

    encoding_matrices = []
    for dim in layer_sizes:
        input_dim = int(next_layer_input.get_shape()[1])

        # Initialize W using random values in interval [-1/sqrt(n) , 1/sqrt(n)]
        W = tf.Variable(tf.random_uniform([input_dim, dim], -1.0 / math.sqrt(input_dim), 1.0 / math.sqrt(input_dim)))

        # Initialize b to zero
        b = tf.Variable(tf.zeros([dim]))

        # We are going to use tied-weights so store the W matrix for later reference.
        encoding_matrices.append(W)

        output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)

        # the input into the next layer is the output of this layer
        next_layer_input = output

    # The fully encoded x value is now stored in the next_layer_input
    encoded_x = next_layer_input

    # build the reconstruction layers by reversing the reductions
    layer_sizes.reverse()
    encoding_matrices.reverse()

    for i, dim in enumerate(layer_sizes[1:] + [int(x.get_shape()[1])]):
        # we are using tied weights, so just lookup the encoding matrix for this step and transpose it
        W = tf.transpose(encoding_matrices[i])
        b = tf.Variable(tf.zeros([dim]))
        output = tf.nn.tanh(tf.matmul(next_layer_input, W) + b)
        next_layer_input = output

    # the fully encoded and reconstructed value of x is here:
    reconstructed_x = next_layer_input

    return {
        'encoded': encoded_x,
        'decoded': reconstructed_x,
        'cost': tf.sqrt(tf.reduce_mean(tf.square(x - reconstructed_x)))
    }


def simple_test():
    sess = tf.Session()
    x = tf.placeholder("float", [None, 4])
    autoencoder = create(x, [2])  #see above in create() method for architecture of autoencoder
    init = tf.initialize_all_variables()
    sess.run(init)
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(autoencoder['cost'])


    # Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
    c1 = np.array([0, 0, 0.5, 0])
    c2 = np.array([0.5, 0, 0, 0])

    # do 1000 training steps
    for i in range(2000):
        # make a batch of 100:
        batch = []
        for j in range(100):
            # pick a random centroid
            if (random.random() > 0.5):
                vec = c1
            else:
                vec = c2
            batch.append(np.random.normal(vec, 0.1))
        sess.run(train_step, feed_dict={x: np.array(batch)})
        if i % 100 == 0:
            print i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch})


def deep_test():
    sess = tf.Session()
    start_dim = 5
    x = tf.placeholder("float", [None, start_dim])
    autoencoder = create(x, [4, 3, 2])
    init = tf.initialize_all_variables()
    sess.run(init)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost'])


    # Our dataset consists of two centers with gaussian noise w/ sigma = 0.1
    c1 = np.zeros(start_dim)
    c1[0] = 1

    print c1

    c2 = np.zeros(start_dim)
    c2[1] = 1

    # do 1000 training steps
    for i in range(5000):
        # make a batch of 100:
        batch = []
        for j in range(1):
            # pick a random centroid
            if (random.random() > 0.5):
                vec = c1
            else:
                vec = c2
            batch.append(np.random.normal(vec, 0.1))
        sess.run(train_step, feed_dict={x: np.array(batch)})
        if i % 100 == 0:
            print i, " cost", sess.run(autoencoder['cost'], feed_dict={x: batch})
            print i, " original", batch[0] # input
            print i, " encoded", sess.run(autoencoder['encoded'], feed_dict={x: batch}) # encoding of last layer
            print i, " decoded", sess.run(autoencoder['decoded'], feed_dict={x: batch}) # decoded input


def deep():
    sess = tf.Session()
    start_dim = 50000  # the starting example was 5. full: 128530. Dimensionality of input. keep as big as possible, but throw singletons away.
    x = tf.placeholder("float", [None, start_dim])
    autoencoder = create(x, [5])  # Dimensionality of the hidden layers. To start with, only use 1 hidden layer.
    init = tf.initialize_all_variables()
    sess.run(init)
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(autoencoder['cost'])

    tokens,vects,norm_tweets = convertTweetsToVec('clinton', start_dim)

    tweets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2, 'clinton')

    vects_dev,norm_tweets_dev = tokenize_tweets.convertTweetsOfficialToVec(start_dim, tokens, tweets_dev)
    devbatch = []
    for v in vects_dev:
        devbatch.append(v)

    sampnr = 12  # which ones of the dev samples to display for sanity check
    print "\noriginal", labels_dev[sampnr], norm_tweets_dev[sampnr]    # print "\noriginal", norm_tweets[2]
    print vects[sampnr]


    # do 1000 training steps
    for i in range(1000):
        # make a batch of 100:
        batch = []
        for j in range(100):
            num = random.randint(0,len(vects)-1)
            batch.append(vects[num])
        sess.run(train_step, feed_dict={x: np.array(batch)})
        if i % 100 == 0:
            decoded = sess.run(autoencoder['decoded'], feed_dict={x: devbatch})  # apply to dev
            encoded = sess.run(autoencoder['encoded'], feed_dict={x: devbatch})  # apply to dev

            dec_tweet = []
            n = 0
            for r in decoded[sampnr]:  # display first result
                if r > 0.1:
                    dec_tweet.append(tokens[n])
                n+=1

            print i, " cost", sess.run(autoencoder['cost'], feed_dict={x: devbatch})
            #print i, " original", batch[0]
            print i, " encoded", encoded[sampnr] # latent representation of input, feed this to SVM(s)
            print i, " decoded", decoded[sampnr]
            print i, " decoded bow", dec_tweet


if __name__ == '__main__':
    deep()


