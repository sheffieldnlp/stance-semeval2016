__author__ = 'Isabelle Augenstein'

from tokenize_tweets import FILE, FILETRUMP
import json
from twokenize_wrapper import tokenize
import io

def readToks():
    tweets = []

    print "Reading official trump data"
    for line in io.open(FILETRUMP, encoding='utf-8', mode='r'):
        if line.startswith('ID\t'):
            continue
        tweet = line.split("\t")[1].replace("\n", "").lower()
        if "#makeamericagreatagain" in tweet or '#votetrump2016' in tweet:
            tweets.append(tweet)
        #if '#trumpistoxic' in tweet or '#trumpno' in tweet or '#bantrump' in tweet:
        #    print tweet
        #for token in tokenize(tweet):
        #    if token == "#makeamericagreatagain":
        #        print line
        #    elif token == '#trumpisdisqualifiedparty':
        #        print line

    print "\nReading general tweets"
    for line in open(FILE, 'r'):
        tweet = json.loads(line)['text'].lower().replace("\n", "")
        if "#makeamericagreatagain" in tweet or '#votetrump2016' in tweet:
            tweets.append(tweet)
        #if '#trumpistoxic' in tweet or '#trumpno' in tweet or '#bantrump' in tweet:
        #    print tweet

    print tweets.__len__()

    f = open("trump_autolabelled.txt", "wb")
    for i, tw in enumerate(tweets):
        f.write(str(i))
        f.write(("\tDonald Trump\t" + tw + "\tFAVOR\n").encode('utf8'))
    f.close()



if __name__ == '__main__':
    readToks()