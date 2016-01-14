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
import itertools


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
def extractFeaturesBOW(tweets, features_final):
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


# train two classifiers, one for on topic/off topic, one for pos/neg
def train_classifiers_TopicVOpinion(feats_train, labels_train, feats_dev, labels_dev, outfilepath):
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
    preds_1_prob = model_1.predict_proba(feats_dev)
    print("Labels topic", labels_dev_tr_1)
    print("Predictions topic", preds_1)
    print("Predictions prob", preds_1_prob)

    model_2 = LogisticRegression(penalty='l1', class_weight='balanced') #, class_weight={1: weight_2} #svm.SVC(class_weight={1: weight})
    model_2.fit(feats_train_2, labels_2)
    preds_2 = model_2.predict(feats_dev)
    preds_2_prob = model_2.predict_proba(feats_dev)
    print("Labels favour/against", labels_dev_tr_2)
    print("Predictions favour/against", preds_2)
    print("Predictions prob", preds_2_prob)

    printPredsToFile_TopicVOpinion(tokenize_tweets.FILEDEV, outfilepath, preds_1, preds_2)


# train two classifiers, one for neutral vs pos, one for neutral vs neg. The same set of training examples are used for both.
def train_classifiers_PosVNeg(feats_train, labels_train, feats_dev, labels_dev, outfilepath):
    labels_1 = []  # this is for the neutral/pos classifier
    labels_2 = []  # this is for the neutral/neg classifier
    labels_dev_tr_1 = [] #transformed from "NONE" etc to 0,1 for neutral/pos classifier
    labels_dev_tr_2 = [] #transformed from "NONE" etc to 0,1 for neutral/neg classifier

    for i, lab in enumerate(labels_train):
        if lab == 'NONE':
            labels_1.append(0)
            labels_2.append(0)
        elif lab == 'FAVOR':
            labels_1.append(1)
            labels_2.append(0)
        elif lab == 'AGAINST':
            labels_1.append(0)
            labels_2.append(1)

    for i, lab in enumerate(labels_dev):
        if lab == 'NONE':
            labels_dev_tr_1.append(0)
            labels_dev_tr_2.append(0)
        elif lab == 'FAVOR':
            labels_dev_tr_1.append(1)
            labels_dev_tr_2.append(0)
        elif lab == 'AGAINST':
            labels_dev_tr_1.append(0)
            labels_dev_tr_2.append(1)



    print("Training classifier...")

    model_1 = LogisticRegression(penalty='l1')#, class_weight='balanced') #svm.SVC(class_weight={1: weight})
    model_1.fit(feats_train, labels_1)
    preds_1 = model_1.predict(feats_dev)
    preds_1_prob = model_1.predict_proba(feats_dev)
    print("Labels neutral/pos", labels_dev_tr_1) # actually this is non-pos/pos
    print("Predictions neutral/pos", preds_1)
    print("Predictions prob", preds_1_prob)

    model_2 = LogisticRegression(penalty='l1') #, class_weight='balanced') #, class_weight={1: weight_2} #svm.SVC(class_weight={1: weight})
    model_2.fit(feats_train, labels_2)
    preds_2 = model_2.predict(feats_dev)
    preds_2_prob = model_2.predict_proba(feats_dev)
    print("Labels neutral/against", labels_dev_tr_2) # actually this is non-neg/neg
    print("Predictions neutral/against", preds_2)
    print("Predictions prob", preds_2_prob)

    printPredsToFile_TopicVOpinion(tokenize_tweets.FILEDEV, outfilepath, preds_1, preds_2)


# train one three-way classifier
def train_classifier_3way(feats_train, labels_train, feats_dev, labels_dev, outfilepath, debug='false', auto_thresh='false'):
    labels = []  # -1 for NONE, 0 for AGAINST, 1 for FAVOR
    labels_dev_tr = [] #transformed from "NONE" etc to -1,0,1

    for i, lab in enumerate(labels_train):
        if lab == 'NONE':
            labels.append(-1)
        elif lab == 'FAVOR':
            labels.append(1)
        elif lab == 'AGAINST':
            labels.append(0)

    for i, lab in enumerate(labels_dev):
        if lab == 'NONE':
            labels_dev_tr.append(-1)
        elif lab == 'FAVOR':
            labels_dev_tr.append(1)
        elif lab == 'AGAINST':
            labels_dev_tr.append(0)


    print("Training classifier...")

    model = LogisticRegression(penalty='l1')#, class_weight='balanced') #svm.SVC(class_weight={1: weight})
    print "Label options", labels
    model.fit(feats_train, labels)
    preds = model.predict(feats_dev)
    preds_prob = model.predict_proba(feats_dev)
    print("Labels", labels_dev_tr)
    print("Predictions", preds)
    print("Predictions prob", preds_prob)

    if auto_thresh == "true":
        print "Number dev samples:\t", len(labels_dev_tr)
        optlabels = optimiseThresh(labels_dev_tr, preds_prob, len(labels_dev_tr)/2)
        printPredsToFileOneModel(tokenize_tweets.FILEDEV, outfilepath, optlabels, len(labels_dev_tr)/2)
    else:
        printPredsToFileOneModel(tokenize_tweets.FILEDEV, outfilepath, preds)

    if debug == "true":
        printProbsToFileOneModel(tokenize_tweets.FILEDEV, outfilepath.replace(".txt", ".debug.txt"), preds_prob, preds)


def getRange(x, y, stepsize):
    ret = []
    while x <= y:
        x += stepsize
        ret.append(x)
    return ret

# optimise threshold for classes on dev set for highest F1.
def optimiseThresh(labels_dev, preds_prob, howmany):

    print "Optimising threshold"

    best_f1 = 0.0
    best_thres = [0.0, 0.0, 0.0]
    # test adding those to probs for none/against/for, hopefully that's fine-grained enough
    for it in getRange(0.0, 1.0, 0.01): #[0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25, 0.3, 0.325, 0.35, 0.375, 0.4]:
         # this produces all possible permutations, e.g. (0, 0, 0), (0, 0, 0.025), (0, 0.025, 0)
        lst = map(list, itertools.product([0.0, it], repeat=3)) # produces list of lists
        del lst[-1] # remove the last element, has same effect as first one
        #print lst
        for perm in lst:
            retlabels, for_p, for_r, for_f1, against_p, against_r, against_f1, macro_f1, a_all, a_tp, a_as_f, a_as_n, f_all, f_tp, f_as_a, f_as_n, n_as_n, n_as_f, n_as_a = computeF1ForThresh(labels_dev[:howmany], preds_prob[:howmany], perm)
            if macro_f1 > best_f1:
                best_f1 = macro_f1
                best_thres = perm


    print "Best thresh", best_thres
    print "Best F1 on dev 1", best_f1


    print "Applying final threshold"

    retlabels, for_p, for_r, for_f1, against_p, against_r, against_f1, macro_f1, a_all, a_tp, a_as_f, a_as_n, f_all, f_tp, f_as_a, f_as_n, n_as_n, n_as_f, n_as_a = computeF1ForThresh(labels_dev[howmany:], preds_prob[howmany:], best_thres)
    print "F1 on dev 2", macro_f1, for_f1, against_f1

    print "\n--------\nConfusion matrix\n--------\n\t\t\t\t\t  Predicted label\n\t\t\t\t\tNeutral\tagainst\tfor"
    print "True label\tNeutral\t", n_as_n, "     ", n_as_a, "     ", n_as_f
    print "True label\tAgainst\t", a_as_n, "     ", a_tp, "     ", a_as_f
    print "True label\tFor\t\t", f_as_n, "     ", f_as_a, "     ", f_tp, "\n--------"

    return retlabels


# get F1 for specific threshold, returns p/r/f1 and confusion matrix
def computeF1ForThresh(devsample, preds_prob, thresh):
    a_tp, a_fp, a_fn, a_all, f_tp, f_fp, f_fn, f_all, a_as_f, a_as_n, f_as_a, f_as_n, n_as_f, n_as_a, n_as_n, n_all = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # save for calculating p/r/f1
    retlabels = []

    # for every dev sample
    for i, l in enumerate(devsample):
        prob = preds_prob[i]  # ith sample

        # check which one is the best label with added ps
        highesti = 0
        highestp = 0.0
        for pi, pl in enumerate(thresh):
            if prob[pi] + pl > highestp: # check if this is better than the previous prediction
                highestp = prob[pi] + pl
                highesti = pi

        # mapping of columns to labels. This depends on the order those labels appear in the training data, here added manually.
        if highesti == 2:
            retlabels.append(-1)
        else:
            retlabels.append(highesti)
        # check is the prediction is correct
        if l == 0: # if the correct one is "against"
            a_all += 1
            if highesti == 0: # correctly predict "against". Reminder: the labels we chose are -1,0,1
                a_tp += 1
            elif highesti == 1: # wrongly predict "for"
                a_as_f += 1
                f_fp += 1
                a_fn += 1
            else: # wrongly predict "neutral"
                a_as_n += 1
                a_fn += 1
        elif l == 1: # if the correct one is "for"
            f_all += 1
            if highesti == 1:
                f_tp += 1
            elif highesti == 0:
                f_as_a += 1
                a_fp += 1
                f_fn += 1
            else:
                f_as_n += 1
                f_fn += 1
        else:
            n_all += 1
            if highesti == 1:
                n_as_f += 1
                f_fp += 1
            elif highesti == 0:
                n_as_a += 1
                a_fp += 1
            else:
                n_as_n += 1
    for_p = f_tp / (f_tp + f_fp + 0.000001)
    for_r = f_tp / (f_all + 0.000001)
    for_f1 = 2 * ((for_p * for_r)/(for_p + for_r + 0.000001))
    against_p = a_tp / (a_tp + a_fp + 0.000001)
    against_r = a_tp / (a_all + 0.000001)
    against_f1 = 2 * ((against_p * against_r)/(against_p + against_r + 0.000001))
    macro_f1 = (for_f1 + against_f1) / 2.0
    print thresh, "\t", macro_f1
    return retlabels, for_p, for_r, for_f1, against_p, against_r, against_f1, macro_f1, a_all, a_tp, a_as_f, a_as_n, f_all, f_tp, f_as_a, f_as_n, n_as_n, n_as_f, n_as_a



# evaluate using the original script, needs to be in same format as train/dev data
def eval(file_gold, file_pred):
    pipe = subprocess.Popen(["perl", "eval.pl", file_gold, file_pred], stdout=sys.stdout) #stdout=subprocess.PIPE)
    pipe.communicate()



# print predictions to file in SemEval format so the official eval script can be applied
def printProbsToFileOneModel(infile, outfile, probs, res):
    outf = open(outfile, 'wb')
    cntr = 0
    for line in io.open(infile, encoding='windows-1252', mode='r'): #for the Trump file it's utf-8
        if line.strip("\n").startswith('ID\t'):
            outf.write(line.strip("\n") + "\tPred\tNONE/AGAINST/FAVOR probs")
        else:
            outl = line.strip("\n").split("\t")
            if res[cntr] == -1:
                outl.append('NONE')
            elif res[cntr] == 0:
                outl.append('AGAINST')
            elif res[cntr] == 1:
                outl.append('FAVOR')
            for p in probs[cntr]:
                outl.append(str(p))
            outf.write("\n" + "\t".join(outl))
            cntr += 1

    outf.close()


# print predictions to file in SemEval format so the official eval script can be applied
def printPredsToFileOneModel(infile, outfile, res, skip=0):
    outf = open(outfile, 'wb')
    cntr = 0
    for line in io.open(infile, encoding='windows-1252', mode='r'): #for the Trump file it's utf-8
        if line.strip("\n").startswith('ID\t'):
            outf.write(line.strip("\n"))
        elif skip > 0:
            skip -= 1
        else:
            outl = line.strip("\n").split("\t")
            if res[cntr] == -1:
                outl[3] = 'NONE'
            elif res[cntr] == 0:
                outl[3] = 'AGAINST'
            elif res[cntr] == 1:
                outl[3] = 'FAVOR'
            outf.write("\n" + "\t".join(outl))
            cntr += 1

    outf.close()


# print predictions to file in SemEval format so the official eval script can be applied
def printPredsToFile_TopicVOpinion(infile, outfile, res_1, res_2):
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


# print predictions to file in SemEval format so the official eval script can be applied
def printPredsToFile_PosVNeg(infile, outfile, res_1, res_2):
    outf = open(outfile, 'wb')
    cntr = 0
    for line in io.open(infile, encoding='windows-1252', mode='r'): #for the Trump file it's utf-8
        if line.strip("\n").startswith('ID\t'):
            outf.write(line.strip("\n"))
        else:
            outl = line.strip("\n").split("\t")
            if res_1[cntr] == 1:
                outl[3] = 'FAVOR'
            elif res_2[cntr] == 1:
                outl[3] = 'AGAINST'
            else:
                outl[3] = 'NONE'
            outf.write("\n" + "\t".join(outl))
            cntr += 1

    outf.close()


if __name__ == '__main__':

    tweets_train, targets_dev, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
    features_final = extractFeatureVocab(tweets_train)
    features_train = extractFeaturesBOW(tweets_train, features_final)

    tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
    features_dev = extractFeaturesBOW(tweets_dev, features_final)

    #train_classifiers_TopicVOpinion(features_train, labels_train, features_dev, labels_dev, "out.txt")
    train_classifier_3way(features_train, labels_train, features_dev, labels_dev, "out_bow_3way.txt", "false", "true")
    #train_classifiers_PosVNeg(features_train, labels_train, features_dev, labels_dev, "out.txt")


    eval(tokenize_tweets.FILEDEV2, "out_bow_3way.txt")