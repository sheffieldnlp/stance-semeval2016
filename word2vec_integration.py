__author__ = 'Isabelle Augenstein'

#!/usr/bin/env python

from gensim.models import word2vec, Phrases
#import pandas as pd
from nltk.corpus import stopwords
from tokenize_tweets import readTweetsOfficial
from twokenize_wrapper import tokenize
import tokenize_tweets
import logging
from tokenize_tweets import readTweets
import re


# prep data for word2vec
def prepData(stopfilter, multiword):
    print "Preparing data..."

    ret = [] # list of lists
    stops = stopwords.words("english")
    # extended with string.punctuation and rt and #semst, removing links further down
    stops.extend(["!", "\"", "#", "$", "%", "&", "\\", "'", "(", ")", "*", "+", ",", "-", ".", "/", ":",
                  ";", "<", "=", ">", "?", "@", "[", "]", "^", "_", "`", "{", "|", "}", "~"])
    stops.extend(["rt", "#semst", "thats", "im", "'s", "...", "via"])

    stops = set(stops)


    print "Reading data..."
    tweets = readTweets()
    tweets_train, targets_train, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
    tweets_trump, targets_trump, labels_trump = readTweetsOfficial(tokenize_tweets.FILETRUMP, 'utf-8', 1)
    print str(len(tweets))
    tweets.extend(tweets_train)
    print str(len(tweets_train)), "\t" , str(len(tweets))
    tweets.extend(tweets_trump)
    print str(len(tweets_trump)), "\t" , str(len(tweets))


    print "Tokenising..."
    for tweet in tweets:
        tokenised_tweet = tokenize(tweet.lower())
        if stopfilter:
            words = [w for w in tokenised_tweet if (not w in stops and not w.startswith("http"))]
            ret.append(words)
        else:
            ret.append(tokenised_tweet)

    if multiword:
        return learnMultiword(ret)
    else:
        return ret


def learnMultiword(ret):
    print "Learning multiword expressions"
    bigram = Phrases(ret)

    print "Sanity checking multiword expressions"
    test = "i like donald trump and hate muslims , go hillary , i like jesus , jesus , against , abortion "
    sent = test.split(" ")
    print bigram[sent]
    return bigram[ret]



def trainWord2VecModel(stopfilter, multiword, modelname):
    tweets = prepData(stopfilter, multiword)
    print "Starting word2vec training"
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # set params
    num_features = 300    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 4       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    trainalgo = 1 # cbow: 0 / skip-gram: 1

    print "Training model..."
    model = word2vec.Word2Vec(tweets, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling, sg = trainalgo)

    # add for memory efficiency
    model.init_sims(replace=True)

    # save the model
    model.save(modelname)


def applyWord2VecModel(modelname):
    model = word2vec.Word2Vec.load(modelname)
    for res in model.most_similar("trump"):
        print res


if __name__ == '__main__':
    #tweets = prepData(True)
    trainWord2VecModel(True, True, "skip_nostop_multi_300features_10minwords_10context")#("300features_40minwords_10context")
    #applyWord2VecModel("skip_nostop_300features_40minwords_10context")

