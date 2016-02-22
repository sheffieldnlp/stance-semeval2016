#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'


import numpy as np
from twokenize_wrapper import tokenize
from training_eval import *
from gensim.models import word2vec, Phrases
from tokenize_tweets import filterStopwords
#from bow_baseline import train_classifier_3way


def extractFeaturesW2V(w2vmodel="skip_nostop_multi_300features_10minwords_10context", phrasemodel="phrase.model", useDev = False):

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

    phmodel = Phrases.load(phrasemodel)
    w2vmodel = word2vec.Word2Vec.load(w2vmodel)
    features_train_w2v = extractW2VAggrFeatures(w2vmodel, phmodel, tweets_train, targets_train, labels_train)
    features_dev_w2v = extractW2VAggrFeatures(w2vmodel, phmodel, tweets_dev, targets_dev, labels_dev)

    return features_train_w2v, labels_train, features_dev_w2v, labels_dev


def extractW2VAggrFeatures(w2vmodel, phrasemodel, tweets, targets, labels):

    feats = []
    # for each tweet, multiply the word vectors
    for i, tweet in enumerate(tweets):
        tokenised_tweet = tokenize(tweet.lower())
        words = filterStopwords(tokenised_tweet)
        numvects = 0
        vect = []
        for token in phrasemodel[words]:
            try:
                s = w2vmodel[token]
                vect.append(s)
                numvects += 1
            except KeyError:
                s = 0.0
        if vect.__len__() > 0:
            mtrmean = np.average(vect, axis=0)
            if i == 0:
                feats = mtrmean
            else:
                feats = np.vstack((feats, mtrmean))
        else:
            feats = np.vstack((feats, np.zeros(300)))  # 300-dimensional vector for now

    return feats


if __name__ == '__main__':

    # get vec for every word/seq, combine
    features_train, labels_train, features_dev, labels_dev = extractFeaturesW2V(useDev = False)

    # train_classifier_3waySGD is another option, for testing elastic net regularisation, doesn't work as well as just l2 though
    train_classifier_3way(features_train, labels_train, features_dev, labels_dev, "out_hillary.txt", [], "false", "false", useDev=False)

    eval(tokenize_tweets.FILEDEV, "out_hillary.txt")
