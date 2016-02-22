__author__ = 'Isabelle Augenstein'

#!/usr/bin/env python

from gensim.models import word2vec, Phrases
#import pandas as pd
from tokenize_tweets import readTweetsOfficial
from twokenize_wrapper import tokenize
import tokenize_tweets
import logging
from tokenize_tweets import readTweets
from tokenize_tweets import KEYWORDS_LONG, KEYWORDS_NEUT, KEYWORDS_NEG, KEYWORDS_POS, filterStopwords





# prep data for word2vec
def prepData(stopfilter, multiword, useDev=False):
    print("Preparing data...")

    ret = [] # list of lists

    print("Reading data...")
    tweets = readTweets()
    tweets_train, targets_train, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
    tweets_trump, targets_trump, labels_trump = readTweetsOfficial(tokenize_tweets.FILETRUMP, 'utf-8', 1)
    print(str(len(tweets)))
    tweets.extend(tweets_train)
    print(str(len(tweets_train)), "\t" , str(len(tweets)))
    tweets.extend(tweets_trump)
    print(str(len(tweets_trump)), "\t" , str(len(tweets)))
    if useDev == True:
        tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
        tweets.extend(tweets_dev)
        print(str(len(tweets_dev)), "\t" , str(len(tweets)))


    print("Tokenising...")
    for tweet in tweets:
        tokenised_tweet = tokenize(tweet.lower())
        if stopfilter:
            words = filterStopwords(tokenised_tweet)
            ret.append(words)
        else:
            ret.append(tokenised_tweet)

    if multiword:
        return learnMultiword(ret)
    else:
        return ret


def learnMultiword(ret):
    print("Learning multiword expressions")
    bigram = Phrases(ret)
    bigram.save("phrase_all.model")

    print("Sanity checking multiword expressions")
    test = "i like donald trump and hate muslims , go hillary , i like jesus , jesus , against , abortion "
    sent = test.split(" ")
    print(bigram[sent])
    return bigram[ret]



def trainWord2VecModel(stopfilter, multiword, modelname):
    tweets = prepData(stopfilter, multiword)
    print("Starting word2vec training")
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # set params
    num_features = 100    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 5          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    trainalgo = 1 # cbow: 0 / skip-gram: 1

    print("Training model...")
    model = word2vec.Word2Vec(tweets, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, sg = trainalgo)

    # add for memory efficiency
    model.init_sims(replace=True)

    # save the model
    model.save(modelname)


# some code for testing
def applyWord2VecModel(modelname):
    model = word2vec.Word2Vec.load(modelname)
    for key in KEYWORDS_LONG['trump']:
        print("\n", key)
        for res in model.most_similar(key, topn=60):
            print(res)

    #for res in model.most_similar("#abortion"):
    #        print res
    #print model.similarity("trump", "muslims")
    #for v in model.vocab:
    #    if "trump" in v.encode('utf-8') or "trump" in v.encode('utf-8'):
    #        print v.encode('utf-8')




# use w2v for extended measuring of targetInTweet
def extractW2VHashFeatures(w2vmodel, phrasemodel, mode, tweets, targets, labels):
    features = []

    inv_topics = {v: k for k, v in tokenize_tweets.TOPICS_LONG.items()}



    for i, tweet in enumerate(tweets):

        # get the neut/pos/neg hashtags
        neut = KEYWORDS_NEUT[inv_topics[targets[i]]]
        pos = KEYWORDS_POS[inv_topics[targets[i]]]
        neg = KEYWORDS_NEG[inv_topics[targets[i]]]

        neutsim = w2vmodel.most_similar(neut, topn=60)
        possim = w2vmodel.most_similar(pos, topn=60)
        negsim = w2vmodel.most_similar(neg, topn=60)

        tokenised_tweet = tokenize(tweet.lower())
        words = filterStopwords(tokenised_tweet)

        neutcnt, poscnt, negcnt, neutsimp, possimp, negsimp = 0, 0, 0, 0, 0, 0


        # transform, as earlier, with the phrase model
        for token in phrasemodel[words]:
            if neut == token:
                neutsimp = 1
            if pos == token:
                possimp = 1
            if neg == token:
                negsimp = 1
            for n, sc in neutsim:
                if sc >= 0.4 and n == token:
                   neutcnt += 1
            for n, sc in possim:
                if sc >= 0.4 and n == token:
                   poscnt += 1
            for n, sc in negsim:
                if sc >= 0.4 and n == token:
                   negcnt += 1

        #print targets[i], "\t", labels[i], "\t", neutcnt, "\t", poscnt, "\t", negcnt, "\t", neutsimp, "\t", possimp, "\t", negsimp
        #featint = [neutcnt, poscnt, negcnt, neutsimp, possimp, negsimp]
        pn = 0
        if possim and negsim:
            pn = 1
            possimp = 0
            negsimp = 0
        if mode == "hash":
            featint = [neutsimp, possimp, negsimp, pn]
            features.append(featint)
        if mode == "w2v_hash":
            featint = [neutcnt, poscnt, negcnt, neutsimp, possimp, negsimp, pn]
            features.append(featint)

    featlabels = []
    if mode == "hash":
        featlabels = ["neut_hash", "pos_hash", "neg_hash", "posneg_hash"]
    if mode == "w2v_hash":
        featlabels = ["neut_extw2v", "pos_extw2v", "neg_extw2v", "neut_hash", "pos_hash", "neg_hash", "posneg_hash"]

    return features, featlabels


# similarity with neut/pos/neg hashtags, not really working
def extractW2VFeaturesSim(w2vmodelfile, phrasemodel, tweets, targets, labels):
    phmodel = Phrases.load(phrasemodel)
    w2vmodel = word2vec.Word2Vec.load(w2vmodelfile)

    inv_topics = {v: k for k, v in tokenize_tweets.TOPICS_LONG.items()}


    for i, tweet in enumerate(tweets):

        # get the neut/pos/neg hashtags
        neut = KEYWORDS_NEUT[inv_topics[targets[i]]]
        pos = KEYWORDS_POS[inv_topics[targets[i]]]
        neg = KEYWORDS_NEG[inv_topics[targets[i]]]

        tokenised_tweet = tokenize(tweet.lower())
        words = filterStopwords(tokenised_tweet)

        neutcnt, poscnt, negcnt = 0, 0, 0
        neutsc, possc, negsc = 0.0, 0.0, 0.0


        # transform, as earlier, with the phrase model
        for token in phmodel[words]:
            try:
                neutsim = w2vmodel.similarity(neut, token)
                neutcnt += 1
                neutsc += neutsim
            except KeyError:
                neutsim = 0
            try:
                possim = w2vmodel.similarity(pos, token)
                possc += possim
                poscnt += 1
            except KeyError:
                possim = 0
            try:
                negsim = w2vmodel.similarity(neg, token)
                negsc += negsim
                negcnt += 1
            except KeyError:
                negsim = 0
            #print targets[i], "\t", token, "\t", neutsim, "\t", possim, "\t", negsim
        neutsc_tweet = neutsc/neutcnt
        possc_tweet = possc/poscnt
        negsc_tweet = negsc/negcnt
        print(targets[i], "\t", labels[i], "\t", neutsc_tweet, "\t", possc_tweet, "\t", negsc_tweet)



# find most similar n words to given word
def applyWord2VecMostSimilar(modelname="skip_nostop_multi_300features_10minwords_10context", word="#abortion", top=20):
    model = word2vec.Word2Vec.load(modelname)
    print("Find ", top, " terms most similar to ", word, "...")
    for res in model.most_similar(word, topn=top):
        print(res)
    print("\n")


# determine similarity between words
def applyWord2VecSimilarityBetweenWords(modelname="skip_nostop_multi_300features_10minwords_10context", w1="trump", w2="muslims"):
    model = word2vec.Word2Vec.load(modelname)
    print("Computing similarity between ", w1, " and ", w2, "...")
    print(model.similarity(w1, w2), "\n")


# search which words/phrases the model knows which contain a searchterm
def applyWord2VecFindWord(modelname="skip_nostop_multi_300features_10minwords_10context", searchterm="trump"):
    model = word2vec.Word2Vec.load(modelname)
    print("Finding terms containing ", searchterm, "...")
    for v in model.vocab:
        if searchterm in v.encode('utf-8'):
            print(v.encode('utf-8'))
    print("\n")


if __name__ == '__main__':
    #tweets = prepData(True, True, True)
    trainWord2VecModel(True, False, "skip_nostop_multi_100features_10minwords_10context")#("300features_40minwords_10context")
    #applyWord2VecModel("skip_nostop_multi_300features_10minwords_10context")


    # Below: simple word2vec test for a trained model with phrases recognised as part of preprocessing

    #applyWord2VecMostSimilar("skip_nostop_multi_300features_10minwords_10context", "#globalwarming", 20) #climatechange #prolife #makeamericagreatagain
    #applyWord2VecSimilarityBetweenWords("skip_nostop_multi_100features_10minwords_5context", "#environment", "#climate")
    #applyWord2VecFindWord("skip_nostop_multi_100features_10minwords_5context", "trump")

