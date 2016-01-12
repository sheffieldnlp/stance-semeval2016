#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'

import tensorflow as tf
import numpy as np
import math
import random
import tokenize_tweets
from tokenize_tweets import convertTweetsToVec, readTweetsOfficial, getTokens
from autoencoder import create
from bow_baseline import train_classifiers, train_classifier, eval, extractFeatures, extractFeatureVocab

# extract autoencoder features based on trained autoencoder model
def extractFeaturesAutoencoder(autoencodermodel, cross_features='false'):
    sess = tf.Session()

    start_dim = 50000

    x = tf.placeholder("float", [None, start_dim])
    autoencoder = create(x, [500])  # Dimensionality of the hidden layers. To start with, only use 1 hidden layer.

    tokens = getTokens(start_dim)

    # read dev data and convert to vectors
    tweets_train, targets_train, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
    vects_train,norm_tweets_train = tokenize_tweets.convertTweetsOfficialToVec(start_dim, tokens, tweets_train)
    vects_train_targets, norm_train_targets = tokenize_tweets.convertTweetsOfficialToVec(start_dim, tokens, targets_train) # optimise runtime with more code later

    # read dev data and convert to vectors
    tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
    vects_dev,norm_tweets_dev = tokenize_tweets.convertTweetsOfficialToVec(start_dim, tokens, tweets_dev)
    vects_dev_targets, norm_dev_targets = tokenize_tweets.convertTweetsOfficialToVec(start_dim, tokens, targets_dev)


    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    # Restore variables from disk.
    saver.restore(sess, autoencodermodel)
    print("Model restored.")

    # apply autoencoder to train and dev data
    encoded_train = sess.run(autoencoder['encoded'], feed_dict={x: vects_train})  # apply to tweets
    encoded_train_target = sess.run(autoencoder['encoded'], feed_dict={x: vects_train_targets})  # apply to target

    encoded_dev = sess.run(autoencoder['encoded'], feed_dict={x: vects_dev})  # apply to tweets
    encoded_dev_target = sess.run(autoencoder['encoded'], feed_dict={x: vects_dev_targets})  # apply to target

    # decoder is just for sanity check, we don't really need that
    #decoded_dev = sess.run(autoencoder['decoded'], feed_dict={x: vects_dev})  # apply to tweets
    #decoded_dev_target = sess.run(autoencoder['decoded'], feed_dict={x: vects_dev_targets})  # apply to target

    print "cost train tweets", sess.run(autoencoder['cost'], feed_dict={x: vects_train})
    print "cost train target", sess.run(autoencoder['cost'], feed_dict={x: vects_train_targets})

    print "cost dev tweets", sess.run(autoencoder['cost'], feed_dict={x: vects_dev})
    print "cost dev target", sess.run(autoencoder['cost'], feed_dict={x: vects_dev_targets})

    features_train = []
    features_dev = []
    if cross_features == "true":
        for i, enc in enumerate(encoded_train_target):
            features_train_i = []
            for v in np.outer(encoded_train[i], encoded_train_target[i]):
                features_train_i.extend(v)
            features_train.append(features_train_i)
        for i, enc in enumerate(encoded_dev_target):
            features_dev_i = []
            for v in np.outer(encoded_dev[i], encoded_dev_target[i]):
                features_dev_i.extend(v)
            features_dev.append(features_dev_i)
    elif cross_features == "added":
        for i, enc in enumerate(encoded_train_target):
            features_train.append(np.append(encoded_train[i], enc))
        for i, enc in enumerate(encoded_dev_target):
            features_dev.append(np.append(encoded_dev[i], enc))
    else:
        features_train = encoded_train
        features_dev = encoded_dev

    print("Features extracted!")

    return features_train, labels_train, features_dev, labels_dev


# extract features autoencoder plus n-gram bow
def extractFeaturesAutoencoderBOW(automodel, cross_features):
    tweets_train, targets_dev, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
    features_final = extractFeatureVocab(tweets_train)
    features_train = extractFeatures(tweets_train, features_final)
    tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
    features_dev = extractFeatures(tweets_dev, features_final)

    features_train_auto, labels_train, features_dev_auto, labels_dev = extractFeaturesAutoencoder(automodel, cross_features)

    # combine features
    for i, featvec in enumerate(features_train_auto):
        features_train[i] = np.append(features_train[i], featvec)  # numpy append works as extend works for python lists
    for i, featvec in enumerate(features_dev_auto):
        features_dev[i] = np.append(features_dev[i], featvec)

    return features_train, labels_train, features_dev, labels_dev


if __name__ == '__main__':
    features_train, labels_train, features_dev, labels_dev = extractFeaturesAutoencoder("model.ckpt", "false")
    #features_train, labels_train, features_dev, labels_dev = extractFeaturesAutoencoderBOW("model.ckpt", "false")

    #train_classifiers(features_train, labels_train, features_dev, labels_dev, "out_auto_added.txt") # train and predict two 2-way models
    train_classifier(features_train, labels_train, features_dev, labels_dev, "out_auto.txt", "true") # train and predict one 3-way model

    eval(tokenize_tweets.FILEDEV, "out_auto.txt") # evaluate with official script