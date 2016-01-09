from sklearn.linear_model import LogisticRegression

__author__ = 'Isabelle'

#!/usr/bin/env python

import numpy as np
from collections import Counter
import io
from tokenize_tweets import convertTweetsToVec, readTweetsOfficial
import tokenize_tweets
from twokenize_wrapper import tokenize
from sklearn import svm


# select features, compile feature vocab
def extractFeatureVocab(tweets, keyword):
    tokens = Counter()
    features_final = []

    tokens_topic = tokenize(tokenize_tweets.TOPICS_LONG[keyword])

    for tweet in tweets:
        tokenised_tweet = tokenize(tweet)
        for token in tokenised_tweet:
            tokens[token] += 1
            for toktopic in tokens_topic:
                tokens[toktopic + '|' + token] += 1
        for l in zip(*[tokenised_tweet[i:] for i in range(2)]):
            tokens["_".join(l)] += 1
            for ltop in zip(*[tokens_topic[i:] for i in range(2)]):
                tokens["_".join(ltop) + '|' + "_".join(l)] += 1

    for token, count in tokens.most_common():
        if count > 1:
            features_final.append(token)
            # print token, count

    return features_final, tokens_topic


# extract BOW n-gram features, returns matrix of features
def extractFeatures(tweets, features_final, tokens_topic):
    matrix = [] # np.zeros((len(features_final), len(tweets)))

    for i, tweet in enumerate(tweets):
        vect = np.zeros((len(features_final)))
        tokenised_tweet = tokenize(tweet)
        for token in tokenised_tweet:
            insertIntoVect(features_final, vect, token)
            for toktopic in tokens_topic:
                insertIntoVect(features_final, vect, toktopic + '|' + token)
        for l in zip(*[tokenised_tweet[i:] for i in range(2)]):
            insertIntoVect(features_final, vect, "_".join(l))
            for ltop in zip(*[tokens_topic[i:] for i in range(2)]):
                insertIntoVect(features_final, vect, "_".join(ltop) + '|' + "_".join(l))

        matrix.append(vect)
        #print " ".join(str(v) for v in vect), "\n"

    return matrix


def insertIntoVect(feats, vect, expr):
    try:
        ind = feats.index(expr)
        vect[ind] = 1
    except (ValueError, IndexError):
        pass
    return vect


def train_classifiers(feats_train, labels_train, feats_dev, labels_dev):
    labels_1 = []  # this is for the topic classifier, will distinguish on/off topic
    labels_2 = []  # this is for the pos/neg classifier
    labels_dev_tr_1 = [] #transformed from "NONE" etc to 0,1 for topic classifier
    feats_train_2 = []

    for i, lab in enumerate(labels_train):
        if lab == 'NONE':
            labels_1.append(0)
        elif lab == 'FAVOR':
            labels_1.append(1)
            labels_2.append(1)
            feats_train_2.append(feats_train[i])
        elif lab == 'AGAINST':
            labels_1.append(1)
            labels_2.append(0)
            feats_train_2.append(feats_train[i])

    for i, lab in enumerate(labels_dev):
        if lab == 'NONE':
            labels_dev_tr_1.append(0)
        elif lab == 'FAVOR':
            labels_dev_tr_1.append(1)
            feats_train_2.append(feats_train[i])
        elif lab == 'AGAINST':
            labels_dev_tr_1.append(1)


    weight = (labels_1.count(1)+labels_1.count(0)+0.0)/labels_1.count(0)

    model_1 = LogisticRegression(penalty='l1', class_weight={1: weight}) #svm.SVC(class_weight={1: weight})
    model_1.fit(feats_train, labels_1)
    preds_1 = model_1.predict(feats_dev)
    print("Labels", labels_dev_tr_1)
    print("Predictions", preds_1)


if __name__ == '__main__':

    tweets_train, labels_train = readTweetsOfficial('clinton', tokenize_tweets.FILETRAIN, 'windows-1252', 2)
    features_final, tokens_topic = extractFeatureVocab(tweets_train, 'clinton')
    features_train = extractFeatures(tweets_train, features_final, tokens_topic)

    tweets_dev, labels_dev = readTweetsOfficial('clinton', tokenize_tweets.FILEDEV, 'windows-1252', 2)
    features_dev = extractFeatures(tweets_dev, features_final, tokens_topic)


    train_classifiers(features_train, labels_train, features_dev, labels_dev)