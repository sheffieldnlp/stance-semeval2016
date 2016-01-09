#!/usr/bin/env python

import json
import io
from collections import defaultdict
import numpy as np
from twokenize_wrapper import tokenize
from token_pb2 import Token, Tokens
from tweet_pb2 import Tweet, Tweets

#FILE = '/home/isabelle/additionalTweetsStanceDetection.json'
FILE = '/Users/Isabelle/Documents/TextualEntailment/SemEvalStance/additionalTweetsStanceDetection_small.json'
#FILE = '/Users/Isabelle/Documents/TextualEntailment/SemEvalStance/additionalTweetsStanceDetection.json'
FILETRAIN = '/Users/Isabelle/Documents/TextualEntailment/SemEvalStance/USFD-StanceDetection/data/semeval/semeval2016-task6-trainingdata.txt'
FILEDEV = '/Users/Isabelle/Documents/TextualEntailment/SemEvalStance/USFD-StanceDetection/data/semeval/semeval2016-task6-trialdata.txt'
FILETRUMP = '/Users/Isabelle/Documents/TextualEntailment/SemEvalStance/USFD-StanceDetection/data/semeval/downloaded_Donald_Trump.txt'

TOKENS = './tokensFinal'

KEYWORDS = {'clinton': ['hillary', 'clinton'],
            'trump' : ['donald trump', 'trump'],
            'climate': 'climate',
            'feminism': ['feminism', 'feminist'],
            'abortion': ['abortion', 'aborting'],
            'atheism': ['atheism', 'atheist']
}

TOPICS_LONG = {'clinton': 'Hillary Clinton',
            'trump' : 'Donald Trump',
            'climate': 'Climate Change is a Real Concern',
            'feminism': 'Feminist Movement',
            'abortion': 'Legalization of Abortion',
            'atheism': 'Atheism'
}


TOPICS = KEYWORDS.keys()

# read tweets from json, get numbers corresponding to tokens from file
def readToks():
    tweets = []
    for line in open(FILE, 'r'):
        tweets.append(json.loads(line))

    tweets_on_topic = defaultdict(list)
    for topic in TOPICS:
        for index, tweet in enumerate(tweets):
            for keyword in KEYWORDS[topic]:
                if keyword in tweet['text'].lower():
                    tweets_on_topic[topic].append(index)
                    break

    tokens_pb = Tokens()
    with open(TOKENS, "rb") as f:
        tokens_pb.ParseFromString(f.read())

    tokens = []
    for token_pb in tokens_pb.tokens:
        if token_pb.count == 1:
            break
        tokens.append(token_pb.token)

    print(str(len(tokens)))
    return tokens,tweets_on_topic,tweets


# read tweets from official files. Change later for unlabelled tweets.
def readTweetsOfficial(topic, tweetfile, encoding, tab):
    tweets = []
    labels = []
    for line in io.open(tweetfile, encoding=encoding, mode='r'):
        if line.startswith('ID\t'):
            continue
        if topic in line.split("\t")[tab-1].lower():
            tweets.append(line.split("\t")[tab])
            labels.append(line.split("\t")[tab+1].strip("\n"))

    return tweets,labels


def writeToksToFile():

    tokens,tweets_on_topic,tweets = readToks()


    for topic in TOPICS:

        tokenized_tweets = Tweets()

        for index in tweets_on_topic[topic]:

            tweet = tweets[index]

            tokenized = tokenized_tweets.tweets.add()
            tokenized.tweet = tweet['text']
            for token in tokenize(tweet['text']):
                try:
                    index = tokens.index(token)
                    tokenized.tokens.append(index)
                except ValueError:
                    tokenized.tokens.append(-1)

            print(tokenized.tokens)
            f = open(topic + '.tweets', "wb")
            f.write(tokenized_tweets.SerializeToString())
            f.close()


def convertTweetsToVec(topic, numtoks):

    tokens,tweets_on_topic,tweets = readToks()

    tokens_sub = tokens[:numtoks]

    tokenized_tweets = Tweets()
    vects = []
    norm_tweets = []

    if topic=='all':
        for topic in TOPICS:
            for index in tweets_on_topic[topic]:

                tweet = tweets[index]
                vect = np.zeros(numtoks)  # dimensionality. the most frequent tokens have a low index, then we can do a cutoff. original: 93988
                norm_tweet = []

                tokenized = tokenized_tweets.tweets.add()
                tokenized.tweet = tweet['text']
                for token in tokenize(tweet['text']):
                    try:
                        index = tokens_sub.index(token)
                    except ValueError:
                        index = -1
                    if index > -1:
                        vect[index] = 1
                        norm_tweet.append(token)
                    else:
                        norm_tweet.append('NULL')

                print(norm_tweet)
                norm_tweets.append(norm_tweet)
                vects.append(vect)
    else:
        for index in tweets_on_topic[topic]:

            tweet = tweets[index]
            vect = np.zeros(numtoks)  # dimensionality. the most frequent tokens have a low index, then we can do a cutoff. original: 93988
            norm_tweet = []

            tokenized = tokenized_tweets.tweets.add()
            tokenized.tweet = tweet['text']
            for token in tokenize(tweet['text']):
                try:
                    index = tokens_sub.index(token)
                except ValueError:
                    index = -1
                if index > -1:
                    vect[index] = 1
                    norm_tweet.append(token)
                else:
                    norm_tweet.append('NULL')

            print(norm_tweet)
            norm_tweets.append(norm_tweet)
            vects.append(vect)

    return tokens_sub,vects,norm_tweets



def convertTweetsOfficialToVec(numtoks, tokens, tweets):

    tokens_sub = tokens[:numtoks]
    tokenized_tweets = Tweets()
    vects = []
    norm_tweets = []

    for tweet in tweets:

        vect = np.zeros(numtoks)  # dimensionality. the most frequent tokens have a low index, then we can do a cutoff. original: 93988
        norm_tweet = []

        tokenized = tokenized_tweets.tweets.add()
        tokenized.tweet = tweet
        for token in tokenize(tweet):
            try:
                index = tokens_sub.index(token)
            except ValueError:
                index = -1
            if index > -1:
                vect[index] = 1
                norm_tweet.append(token)
            else:
                norm_tweet.append('NULL')

        print(norm_tweet)
        norm_tweets.append(norm_tweet)
        vects.append(vect)


    return vects,norm_tweets



if __name__ == '__main__':
    #writeToksToFile()
    #convertTweetsToVec('climate', 5000)
    #readTweetsOfficial()
    readToks()