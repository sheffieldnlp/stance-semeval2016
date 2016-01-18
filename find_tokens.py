#!/usr/bin/env python

import json
from collections import Counter
from twokenize_wrapper import tokenize
from token_pb2 import Token, Tokens
import tokenize_tweets
import io
from nltk.corpus import stopwords
from word2vec_integration import filterStopwords
from gensim.models import word2vec, Phrases



#INPUT = '/home/isabelle/additionalTweetsStanceDetection.json'
INPUT = '/Users/Isabelle/Documents/TextualEntailment/SemEvalStance/stanceDetection.json'
#INPUT = 'stanceDetection.json'
#INPUT = '/Users/Isabelle/Documents/TextualEntailment/SemEvalStance/additionalTweetsStanceDetection_small.json'
#INPUT = '/Users/Isabelle/Documents/TextualEntailment/SemEvalStance/USFD-StanceDetection/data/semeval/downloaded_Donald_Trump.txt'
#OUTPUT = './tokensFinal'  # redownload this, changed on 15 January
OUTPUT = './tokensPhrases'

# tokenise the collected tweets
def findTokensJson():
    tokens = Counter()

    for line in open(INPUT, 'r'):
        for token in tokenize(json.loads(line)['text']):
            tokens[token] += 1

    output = open(OUTPUT, "wb")
    tokens_pb = Tokens()

    for token, count in tokens.most_common():
        token_pb = tokens_pb.tokens.add()
        token_pb.token = token
        token_pb.count = count

    output.write(tokens_pb.SerializeToString())
    output.close




# tokenise the collected tweets, plus all the other ones, process with phrases model, for training autoencoder
def findTokensPhrases(phrasemodel="phrase.model"):
    tokencnt = Counter()
    bigram = Phrases(phrasemodel)

    twcntr = 0
    supercntr = 0
    trumpcntr = 0

    for line in open(INPUT, 'r'):
        twcntr += 1
        tokenised = tokenize(json.loads(line)['text'].lower())
        tokens = filterStopwords(tokenised) # filter stopwords
        for token in bigram[tokens]: # calling the phrase model, this leaves some as single tokens and feq occurring ones as bigrams
            tokencnt[token] += 1


    for line in io.open(tokenize_tweets.FILETRAIN, encoding='windows-1252', mode='r'): #for the Trump file it's utf-8
        if line.startswith('ID\t'):
            continue
        tokens = filterStopwords(tokenize(line.split("\t")[2].lower()))  #For Trump it's [1]
        for token in bigram[tokens]:
            supercntr += 1
            tokencnt[token] += 1

    for line in io.open(tokenize_tweets.FILETRUMP, encoding='utf-8', mode='r'): #for the Trump file it's utf-8
        if line.startswith('ID\t'):
            continue
        tokens = filterStopwords(tokenize(line.split("\t")[1].lower()))  #For Trump it's [1]
        for token in bigram[tokens]:
            trumpcntr += 1
            tokencnt[token] += 1


    output = open(OUTPUT, "wb")
    tokens_pb = Tokens()

    for token, count in tokencnt.most_common():
        if count > 1:  # not even worth saving singletons
            token_pb = tokens_pb.tokens.add()
            token_pb.token = token
            token_pb.count = count

    print "Saving token counts for ", tokencnt.__sizeof__(), ". ", twcntr, " unlabelled tweets, ", trumpcntr, " Donald Trump tweets, ", supercntr, " labelled tweets"

    output.write(tokens_pb.SerializeToString())
    output.close





# tokenise the collected tweets, plus all the other ones, for training autoencoder
def findTokensAll():
    tokens = Counter()

    twcntr = 0
    supercntr = 0
    trumpcntr = 0

    for line in open(INPUT, 'r'):
        twcntr += 1
        for token in tokenize(json.loads(line)['text']):
            tokens[token] += 1

    for line in io.open(tokenize_tweets.FILETRAIN, encoding='windows-1252', mode='r'): #for the Trump file it's utf-8
        if line.startswith('ID\t'):
            continue
        for token in tokenize(line.split("\t")[2]):  #For Trump it's [1]
            supercntr += 1
            tokens[token] += 1

    for line in io.open(tokenize_tweets.FILETRUMP, encoding='utf-8', mode='r'): #for the Trump file it's utf-8
        if line.startswith('ID\t'):
            continue
        for token in tokenize(line.split("\t")[1]):  #For Trump it's [1]
            trumpcntr += 1
            tokens[token] += 1


    output = open(OUTPUT, "wb")
    tokens_pb = Tokens()

    for token, count in tokens.most_common():
        if count > 1:  # not even worth saving singletons
            token_pb = tokens_pb.tokens.add()
            token_pb.token = token
            token_pb.count = count

    print "Saving token counts for ", tokens.__sizeof__(), ". ", twcntr, " unlabelled tweets, ", trumpcntr, " Donald Trump tweets, ", supercntr, " labelled tweets"

    output.write(tokens_pb.SerializeToString())
    output.close


# tokenise the official data
def findTokensOfficial():
    tokens = Counter()

    for line in io.open(INPUT, encoding='windows-1252', mode='r'): #for the Trump file it's utf-8
        if line.startswith('ID\t'):
            continue
        for token in tokenize(line.split("\t")[2]):  #For Trump it's [1]
            tokens[token] += 1

    output = open(OUTPUT, "wb")
    tokens_pb = Tokens()

    for token, count in tokens.most_common():
        token_pb = tokens_pb.tokens.add()
        token_pb.token = token
        token_pb.count = count

    output.write(tokens_pb.SerializeToString())
    output.close


if __name__ == '__main__':
    #findTokensJson() #this is to tokenise the unlabelled tweets, needs to be run first. OUTPUT = './tokens
    #findTokensOfficial() #this is to tokenise the labelled tweets, needs to be run first. OUTPUT = './tokensOfficialDev, ./tokensOfficialTrain, ./tokensOfficialTrump

    #findTokensAll()
    findTokensPhrases()