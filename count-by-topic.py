#!/usr/bin/env python

import json
import sys
from collections import defaultdict
from twokenize_wrapper import tokenize
from token_pb2 import Token, Tokens
from tweet_pb2 import Tweet, Tweets

#TWEETS = './small.tweets'
TWEETS = './all.tweets'
TOKENS = './tokens'

keywords = {'clinton': ['hillary', 'clinton'], 
            'obama' : ['barack', 'obama'],
            'climate': ['climate'],
            'feminism': ['feminism', 'feminist'],
            'abortion': ['abortion', 'aborting'],
            'atheism': ['atheism', 'atheist']
}

topics = keywords.keys()

tokens_pb = Tokens()
with open(TOKENS, "rb") as f:
    tokens_pb.ParseFromString(f.read())

tokens = []
for token_pb in tokens_pb.tokens:
    if token_pb.count == 1:
        break
    tokens.append(token_pb.token)

print len(tokens)

sys.exit()

tweets_on_topic = defaultdict(list)
for topic in topics:
    for index, tweet in enumerate(tweets):
        for keyword in keywords[topic]:
            if keyword in tweet['text'].lower():
                tweets_on_topic[topic].append(index)
                break


for topic in topics:

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

        f = open(topic + '.tweets', "wb")
        f.write(tokenized_tweets.SerializeToString())
        f.close()
