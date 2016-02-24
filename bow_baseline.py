#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'


import numpy as np
from collections import Counter
import tokenize_tweets
from twokenize_wrapper import tokenize
from tokenize_tweets import readTweetsOfficial
from training_eval import *
from affect import getAffect
from autoencoder_eval import extractFeaturesCrossTweetTarget, extractFeaturesAutoencoder
from emoticons import analyze_tweet
from word2vec_integration import extractW2VHashFeatures
from gensim.models import word2vec, Phrases
from tokenize_tweets import filterStopwords


# select features, compile feature vocab
def extractFeatureVocab(tweets, keyword="all", usephrasemodel=True, phrasemodel="phrase.model", anon_targets=False):
    tokencounts = Counter()
    features_final = []
    bigram = Phrases(phrasemodel)
    #tokens_topic = []

    #if keyword == "all":
    #    for top in tokenize_tweets.TOPICS:
    #        if top != 'clinton':
    #            for tok in tokenize(tokenize_tweets.TOPICS_LONG[top]):
    #                tokens_topic.append(tok)
    #else:
    #    tokens_topic = tokenize(tokenize_tweets.TOPICS_LONG[keyword])

    for tweet in tweets:
        if usephrasemodel == False:
            tokenised_tweet = tokenize(tweet)
            for token in tokenised_tweet:  #unigram features
                tokencounts[token] += 1
                #for toktopic in tokens_topic:
                #    tokencounts[toktopic + '|' + token] += 1
            for l in zip(*[tokenised_tweet[i:] for i in range(2)]): #bigram features
                tokencounts["_".join(l)] += 1
                #for ltop in zip(*[tokens_topic[i:] for i in range(2)]):
                #    tokencounts["_".join(ltop) + '|' + "_".join(l)] += 1
        else:
            # this includes unigrams and frequent bigrams
            tokens = filterStopwords(tokenize(tweet.lower()))  #For Trump it's [1]
            phrasetoks = bigram[tokens]
            target_keywords = []
            if anon_targets==True:
                for top in tokenize_tweets.TOPICS:
                    if top == "climate": # hack, this is the only non-list value
                        target_keywords.append("climate")
                    else:
                        #for keyw in tokenize_tweets.KEYWORDS[top]:
                        target_keywords.extend(tokenize_tweets.KEYWORDS[top])

                phrasetoks_new = []
                for token in phrasetoks:
                    for keyw in target_keywords:
                        if keyw in token:
                            token = token.replace(keyw, "TARGET")
                    phrasetoks_new.append(token)
                phrasetoks = phrasetoks_new

            for token in phrasetoks:
                tokencounts[token] += 1
            for l in zip(*[phrasetoks[i:] for i in range(2)]):
                tokencounts["_".join(l)] += 1

    for token, count in tokencounts.most_common():
        if count > 1:
            features_final.append(token)
            #print token, count

    return features_final


# extract BOW n-gram features, returns matrix of features
def extractFeaturesBOW(tweets, targets, features_final, anon_targets=False, usephrasemodel=False, phrasemodel="phrase.model"):

    bigram = Phrases(phrasemodel)

    matrix = [] # np.zeros((len(features_final), len(tweets)))

    for i, tweet in enumerate(tweets):
        vect = np.zeros((len(features_final)))
        if usephrasemodel == False:
            tokenised_tweet = tokenize(tweet)
            for token in tokenised_tweet:
                insertIntoVect(features_final, vect, token)
                #for toktopic in tokens_topic:
                #    insertIntoVect(features_final, vect, toktopic + '|' + token)
            for l in zip(*[tokenised_tweet[i:] for i in range(2)]):
                insertIntoVect(features_final, vect, "_".join(l))
                #for ltop in zip(*[tokens_topic[i:] for i in range(2)]):
                #    insertIntoVect(features_final, vect, "_".join(ltop) + '|' + "_".join(l))
        else:
            inv_topics = {v: k for k, v in tokenize_tweets.TOPICS_LONG.items()}
            target_keywords = tokenize_tweets.KEYWORDS.get(inv_topics.get(targets[i]))

            tokens = filterStopwords(tokenize(tweet.lower()))  #For Trump it's [1]
            phrasetoks = bigram[tokens]


            if anon_targets==True:
                phrasetoks_new = []
                for token in phrasetoks:
                    if target_keywords == "climate":
                        if target_keywords in token:
                            token = token.replace(keyw, "TARGET")
                    else:
                        for keyw in target_keywords:
                            if keyw in token:
                                token = token.replace(keyw, "TARGET")
                    phrasetoks_new.append(token)
                phrasetoks = phrasetoks_new

            for token in phrasetoks:
                insertIntoVect(features_final, vect, token)
            for l in zip(*[phrasetoks[i:] for i in range(2)]):
                insertIntoVect(features_final, vect, "_".join(l))

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


# extract emoticon features
def extractEmoticons(tweets):
    vects = [] # BOTH_HS, HAPPY, SAD, NA
    vocab = ["BOTH_HS", "HAPPY", "SAD", "NA"]
    for i, tweet in enumerate(tweets):
        vect = np.zeros(4)
        emo = analyze_tweet(tweet)
        if emo == "NA":
            vect[0] = 1
        elif emo == "HAPPY":
            vect[1] = 1
        elif emo == "SAD":
            vect[2] = 1
        elif emo == "BOTH_HS":
            vect[3] = 1
        vects.append(vect)
    return vects, vocab



# extract features autoencoder plus n-gram bow
def extractFeaturesMulti(features=["auto_false", "bow", "targetInTweet", "emoticons", "affect", "w2v", "bow_phrase"]
        , automodel="model.ckpt", w2vmodel="skip_nostop_multi_300features_10minwords_10context", phrasemodel="phrase.model",
        useDev=True):
    if useDev==False:
        tweets_train, targets_train, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
        tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
    else:
        tweets_train, targets_train, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
        tweets_origdev, targets_origdev, labels_origdev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
        tweets_train.extend(tweets_origdev)
        targets_train.extend(targets_origdev)
        labels_train.extend(labels_origdev)
        tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILETEST, 'windows-1252', 2)

    features_final = []

    if features.__contains__("bow"):
        features_final = extractFeatureVocab(tweets_train)
        features_train = extractFeaturesBOW(tweets_train, targets_train, features_final)
        features_dev = extractFeaturesBOW(tweets_dev, targets_dev, features_final)
    elif features.__contains__("targetInTweet"):
        features_train = extractFeaturesCrossTweetTarget(tweets_train, targets_train)
        features_dev = extractFeaturesCrossTweetTarget(tweets_dev, targets_dev)
        features_final.append("targetInTweet")

    if features.__contains__("bow_phrase") or features.__contains__("bow_phrase_anon"):
        if features.__contains__("bow_phrase"):
            features_vocab = extractFeatureVocab(tweets_train, usephrasemodel=True)
            features_train_phrbow = extractFeaturesBOW(tweets_train, targets_train, features_vocab, usephrasemodel=True)
            features_dev_phrbow = extractFeaturesBOW(tweets_dev, targets_dev, features_vocab, usephrasemodel=True)
        elif features.__contains__("bow_phrase_anon"):
            features_vocab = extractFeatureVocab(tweets_train, usephrasemodel=True, anon_targets=True)
            features_train_phrbow = extractFeaturesBOW(tweets_train, targets_train, features_vocab, usephrasemodel=True, anon_targets=True)
            features_dev_phrbow = extractFeaturesBOW(tweets_dev, targets_dev, features_vocab, usephrasemodel=True, anon_targets=True)
        features_final.extend(features_vocab)

    if features.__contains__("auto_added"):
        useph=False
        if "phrase" in automodel:
            useph=True
        features_train_auto, labels_train, features_dev_auto, labels_dev = extractFeaturesAutoencoder(automodel, tweets_train, targets_train, labels_train, tweets_dev, targets_dev, labels_dev, "added", usephrasemodel=useph)
    elif features.__contains__("auto_true"):
        useph=False
        if "phrase" in automodel:
            useph=True
        features_train_auto, labels_train, features_dev_auto, labels_dev = extractFeaturesAutoencoder(automodel, tweets_train, targets_train, labels_train, tweets_dev, targets_dev, labels_dev, "true", usephrasemodel=useph)
    elif features.__contains__("auto_false"):
        useph=False
        if "phrase" in automodel:
            useph=True
        features_train_auto, labels_train, features_dev_auto, labels_dev = extractFeaturesAutoencoder(automodel, tweets_train, targets_train, labels_train, tweets_dev, targets_dev, labels_dev, "false", usephrasemodel=useph)

    targetInTweetTrain = []
    targetInTweetDev = []
    if features.__contains__("targetInTweet") and features.__contains__("bow"):
        targetInTweetTrain = extractFeaturesCrossTweetTarget(tweets_train, targets_train)
        targetInTweetDev = extractFeaturesCrossTweetTarget(tweets_dev, targets_dev)
        features_final.append("targetInTweet")
    if features.__contains__("emoticons"):
        emoticons_train, emoticons_vocab = extractEmoticons(tweets_train)
        emoticons_dev, emoticons_vocab = extractEmoticons(tweets_dev)
        for emo in emoticons_vocab:
            features_final.append("Emoticon_" + emo)
    if features.__contains__("affect"):
        affect_train, affect_vocab = getAffect(tweets_train)
        affect_dev, affect_vocab = getAffect(tweets_dev)
        for aff in affect_vocab:
            features_final.append("WNaffect_" + aff)

    if features.__contains__("hash"):
        phmodel = Phrases.load(phrasemodel)
        w2vmodel = word2vec.Word2Vec.load(w2vmodel)
        features_train_w2v, features_w2v_vocab = extractW2VHashFeatures(w2vmodel, phmodel, "hash", tweets_train, targets_train, labels_train)
        features_dev_w2v, features_w2v_vocab = extractW2VHashFeatures(w2vmodel, phmodel, "hash", tweets_dev, targets_dev, labels_dev)
    elif features.__contains__("w2v_hash"): # this contains hash
        phmodel = Phrases.load(phrasemodel)
        w2vmodel = word2vec.Word2Vec.load(w2vmodel)
        features_train_w2v, features_w2v_vocab = extractW2VHashFeatures(w2vmodel, phmodel, "w2v_hash", tweets_train, targets_train, labels_train)
        features_dev_w2v, features_w2v_vocab = extractW2VHashFeatures(w2vmodel, phmodel, "w2v_hash", tweets_dev, targets_dev, labels_dev)

    # combine features
    for i, featvec in enumerate(features_train):#features_train_auto)
        if features.__contains__("auto_added") or features.__contains__("auto_true") or features.__contains__("auto_false"):
            features_train[i] = np.append(features_train[i], features_train_auto[i])  # numpy append works as extend works for python lists
        if features.__contains__("targetInTweet") and features.__contains__("bow"):
            features_train[i] = np.append(features_train[i], targetInTweetTrain[i])
        if features.__contains__("bow_phrase") or features.__contains__("bow_phrase_anon"):
            features_train[i] = np.append(features_train[i], features_train_phrbow[i])
        if features.__contains__("emoticons"):
            features_train[i] = np.append(features_train[i], emoticons_train[i])
        if features.__contains__("affect"):
            features_train[i] = np.append(features_train[i], affect_train[i])
        if features.__contains__("w2v_hash") or features.__contains__("hash"):
            features_train[i] = np.append(features_train[i], features_train_w2v[i])
    for i, featvec in enumerate(features_dev):#features_dev_auto):
        if features.__contains__("auto_added") or features.__contains__("auto_true") or features.__contains__("auto_false"):
            features_dev[i] = np.append(features_dev[i], features_dev_auto[i])
        if features.__contains__("targetInTweet") and features.__contains__("bow"):
            features_dev[i] = np.append(features_dev[i], targetInTweetDev[i])
        if features.__contains__("bow_phrase") or features.__contains__("bow_phrase_anon"):
            features_dev[i] = np.append(features_dev[i], features_dev_phrbow[i])
        if features.__contains__("emoticons"):
            features_dev[i] = np.append(features_dev[i], emoticons_dev[i])
        if features.__contains__("affect"):
            features_dev[i] = np.append(features_dev[i], affect_dev[i])
        if features.__contains__("w2v_hash") or features.__contains__("hash"):
            features_dev[i] = np.append(features_dev[i], features_dev_w2v[i])


    return features_train, labels_train, features_dev, labels_dev, features_final




if __name__ == '__main__':

    # Options: "auto_false", "bow", "targetInTweet", "emoticons", "affect", "w2v", "hash", "bow_phrase"
    features_train, labels_train, features_dev, labels_dev, feature_vocab = extractFeaturesMulti(["auto_false", "targetInTweet"],
        "model_phrase_100_samp500_it2000.ckpt", useDev=False)#  "model_trump_phrase_100_samp500_it2600.ckpt")#"model_100_samp500.ckpt")

    #train_classifiers_TopicVOpinion(features_train, labels_train, features_dev, labels_dev, "out.txt")

    # train_classifier_3waySGD is another option, for testing elastic net regularisation, doesn't work as well as just l2 though
    train_classifier_3way(features_train, labels_train, features_dev, labels_dev, "out_hillary_bow_targetInTweet.txt", feature_vocab, "false", "false", useDev=False)
    #train_classifiers_PosVNeg(features_train, labels_train, features_dev, labels_dev, "out.txt")


    eval(tokenize_tweets.FILEDEV, "out_hillary_bow_targetInTweet.txt")