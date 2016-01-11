#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'


import numpy as np
from collections import Counter
import io
from tokenize_tweets import convertTweetsToVec, readTweetsOfficial
import tokenize_tweets
from twokenize_wrapper import tokenize
from sklearn import svm
import subprocess
import sys
from sklearn.linear_model import LogisticRegression


# select features, compile feature vocab
def extractFeatureVocab(tweets, keyword="all"):
    tokens = Counter()
    features_final = []
    #tokens_topic = []

    #if keyword == "all":
    #    for top in tokenize_tweets.TOPICS:
    #        if top != 'clinton':
    #            for tok in tokenize(tokenize_tweets.TOPICS_LONG[top]):
    #                tokens_topic.append(tok)
    #else:
    #    tokens_topic = tokenize(tokenize_tweets.TOPICS_LONG[keyword])

    for tweet in tweets:
        tokenised_tweet = tokenize(tweet)
        for token in tokenised_tweet:  #unigram features
            tokens[token] += 1
            #for toktopic in tokens_topic:
            #    tokens[toktopic + '|' + token] += 1
        for l in zip(*[tokenised_tweet[i:] for i in range(2)]): #bigram features
            tokens["_".join(l)] += 1
            #for ltop in zip(*[tokens_topic[i:] for i in range(2)]):
            #    tokens["_".join(ltop) + '|' + "_".join(l)] += 1

    for token, count in tokens.most_common():
        if count > 1:
            features_final.append(token)
            # print token, count

    return features_final


# extract BOW n-gram features, returns matrix of features
def extractFeatures(tweets, features_final):
    matrix = [] # np.zeros((len(features_final), len(tweets)))

    for i, tweet in enumerate(tweets):
        vect = np.zeros((len(features_final)))
        tokenised_tweet = tokenize(tweet)
        for token in tokenised_tweet:
            insertIntoVect(features_final, vect, token)
            #for toktopic in tokens_topic:
            #    insertIntoVect(features_final, vect, toktopic + '|' + token)
        for l in zip(*[tokenised_tweet[i:] for i in range(2)]):
            insertIntoVect(features_final, vect, "_".join(l))
            #for ltop in zip(*[tokens_topic[i:] for i in range(2)]):
            #    insertIntoVect(features_final, vect, "_".join(ltop) + '|' + "_".join(l))

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


def train_classifiers(feats_train, labels_train, feats_dev, labels_dev, outfilepath):
    labels_1 = []  # this is for the topic classifier, will distinguish on/off topic
    labels_2 = []  # this is for the pos/neg classifier
    labels_dev_tr_1 = [] #transformed from "NONE" etc to 0,1 for topic classifier
    labels_dev_tr_2 = [] #transformed from "NONE" etc to -1,0,1 for pos/neg classifier
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
            labels_dev_tr_2.append(-1)
        elif lab == 'FAVOR':
            labels_dev_tr_1.append(1)
            labels_dev_tr_2.append(1)
        elif lab == 'AGAINST':
            labels_dev_tr_1.append(1)
            labels_dev_tr_2.append(0)

    #weight_1 = (labels_1.count(1)+0.0)/labels_1.count(0)
    #weight_2 = (labels_2.count(1)+0.0)/labels_2.count(0)

    print("Training classifier...")

    model_1 = LogisticRegression(penalty='l1', class_weight='balanced') #svm.SVC(class_weight={1: weight})
    model_1.fit(feats_train, labels_1)
    preds_1 = model_1.predict(feats_dev)
    print("Labels topic", labels_dev_tr_1)
    print("Predictions topic", preds_1)

    model_2 = LogisticRegression(penalty='l1', class_weight='balanced') #, class_weight={1: weight_2} #svm.SVC(class_weight={1: weight})
    model_2.fit(feats_train_2, labels_2)
    preds_2 = model_2.predict(feats_dev)
    print("Labels favour/against", labels_dev_tr_2)
    print("Predictions favour/against", preds_2)

    printPredsToFile(tokenize_tweets.FILEDEV, outfilepath, preds_1, preds_2)


# evaluate using the original script, needs to be in same format as train/dev data
def eval(file_gold, file_pred):
    pipe = subprocess.Popen(["perl", "eval.pl", file_gold, file_pred], stdout=sys.stdout) #stdout=subprocess.PIPE)
    pipe.communicate()


# print predictions to file in SemEval format so the official eval script can be applied
def printPredsToFile(infile, outfile, res_1, res_2):
    outf = open(outfile, 'wb')
    cntr = 0
    for line in io.open(infile, encoding='windows-1252', mode='r'): #for the Trump file it's utf-8
        if line.strip("\n").startswith('ID\t'):
            outf.write(line.strip("\n"))
        else:
            outl = line.strip("\n").split("\t")
            if res_1[cntr] == 0:
                outl[3] = 'NONE'
            elif res_2[cntr] == 0:
                outl[3] = 'AGAINST'
            elif res_2[cntr] == 1:
                outl[3] = 'FAVOR'
            outf.write("\n" + "\t".join(outl))
            cntr += 1

    outf.close()


if __name__ == '__main__':

    tweets_train, targets_dev, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
    features_final = extractFeatureVocab(tweets_train)
    features_train = extractFeatures(tweets_train, features_final)

    tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
    features_dev = extractFeatures(tweets_dev, features_final)

    train_classifiers(features_train, labels_train, features_dev, labels_dev, "out.txt")

    eval(tokenize_tweets.FILEDEV, "out.txt")