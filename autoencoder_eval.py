#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'

import tensorflow as tf
import numpy as np
from training_eval import train_classifier_3way, eval
import tokenize_tweets
from tokenize_tweets import readTweetsOfficial, readToks2
from autoencoder import create
from twokenize_wrapper import tokenize


# extract autoencoder features based on trained autoencoder model
def extractFeaturesAutoencoder(autoencodermodel, tweets_train, targets_train, labels_train, tweets_dev, targets_dev, labels_dev,
                               cross_features='false', usephrasemodel=False):
    sess = tf.Session()

    start_dim = 50000

    x = tf.placeholder("float", [None, start_dim])
    autoencoder = create(x, [100])  # Dimensionality of the hidden layers. To start with, only use 1 hidden layer.

    tokens = readToks2(start_dim, usephrasemodel)

    # read dev data and convert to vectors
    #tweets_train, targets_train, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
    vects_train,norm_tweets_train = tokenize_tweets.convertTweetsOfficialToVec(start_dim, tokens, tweets_train, filtering=True)
    vects_train_targets, norm_train_targets = tokenize_tweets.convertTweetsOfficialToVec(start_dim, tokens, targets_train) # optimise runtime with more code later

    # read dev data and convert to vectors
    #tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
    vects_dev,norm_tweets_dev = tokenize_tweets.convertTweetsOfficialToVec(start_dim, tokens, tweets_dev, filtering=True)
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

    print("cost train tweets", sess.run(autoencoder['cost'], feed_dict={x: vects_train}))
    print("cost train target", sess.run(autoencoder['cost'], feed_dict={x: vects_train_targets}))

    print("cost dev tweets", sess.run(autoencoder['cost'], feed_dict={x: vects_dev}))
    print("cost dev target", sess.run(autoencoder['cost'], feed_dict={x: vects_dev_targets}))

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






# "target in tweet" feature extraction
def extractFeaturesCrossTweetTarget(tweets, targets):
    ret = []
    inv_topics = {v: k for k, v in tokenize_tweets.TOPICS_LONG.items()}
    #TOPICS = inv_topics.keys()
    for i, tweet in enumerate(tweets):
        tokenised_tweet = tokenize(tweet)
        target_keywords = tokenize_tweets.KEYWORDS.get(inv_topics.get(targets[i]))
        target_in_tweet = 0
        for key in target_keywords:
            if tweet.__contains__(key):
                target_in_tweet = 1
                break
        # option below cares for tokenisation, but since hashtags are not tokenised at the moment, the above works better
        #for tweettok in tokenised_tweet:
        #    if tweettok in target_keywords:
        #        target_in_tweet = 1
        #        break
        ret.append(target_in_tweet)
    return ret





if __name__ == '__main__':
    useDev = True
    if useDev == False:
        tweets_train, targets_train, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
        tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
    else:
        tweets_train, targets_train, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
        tweets_origdev, targets_origdev, labels_origdev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
        tweets_train.extend(tweets_origdev)
        targets_train.extend(targets_origdev)
        labels_train.extend(labels_origdev)
        tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILETEST, 'windows-1252', 2)

     # "model_phrase_100_samp500_it2000.ckpt"
    features_train, labels_train, features_dev, labels_dev = extractFeaturesAutoencoder("model_trump_phrase_100_samp500_it2600.ckpt",
            tweets_train, targets_train, labels_train, tweets_dev, targets_dev, labels_dev, "false", True)

    #train_classifiers(features_train, labels_train, features_dev, labels_dev, "out_auto_added.txt") # train and predict two 2-way models
    train_classifier_3way(features_train, labels_train, features_dev, labels_dev, "out_trump_postprocess.txt", [], "false", "false", useDev=useDev)
    #train_classifiers_PosVNeg(features_train, labels_train, features_dev, labels_dev, "out_auto.txt")


    eval(tokenize_tweets.FILETEST, "out_trump_postprocess.txt")