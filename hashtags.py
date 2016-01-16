__author__ = 'Isabelle Augenstein'

import tokenize_tweets
from twokenize_wrapper import tokenize
from tokenize_tweets import readTweetsOfficial
from collections import Counter


# select features, compile feature vocab
def countHashTags(tweets, labels):
    neut = Counter()
    neg = Counter()
    pos = Counter()
    all = Counter()

    for it, tweet in enumerate(tweets):
        tokenised_tweet = tokenize(tweet)
        label = labels[it]
        for token in tokenised_tweet:
            if token.startswith("#"):
                all[token] += 1
                if label == "NONE":
                    neut[token] += 1
                elif label == "AGAINST":
                    neg[token] += 1
                elif label == "FAVOR":
                    pos[token] += 1

    print "Hashtags\tAll\tNeut\tNeg\tPos"
    for token, count in all.most_common():
        neutrcnt, poscnt, negcnt = 0, 0, 0
        if neut.__contains__(token):
            neutrcnt = neut[token]
        if neg.__contains__(token):
            negcnt = neg[token]
        if pos.__contains__(token):
            poscnt = pos[token]
        print token, "\t", count, "\t", neutrcnt, "\t", negcnt, "\t", poscnt



if __name__ == '__main__':
    #tweets_train, targets_dev, labels_train = readTweetsOfficial(tokenize_tweets.FILETRAIN, 'windows-1252', 2)
    #tweets_train, targets_dev, labels_train = readTweetsOfficial(tokenize_tweets.FILETRUMP, 'utf-8', 1)
    tweets_train, targets_dev, labels_train = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
    countHashTags(tweets_train, labels_train)

