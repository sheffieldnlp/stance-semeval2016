from tokenize_tweets import readTweetsOfficial
import tokenize_tweets
import io
from training_eval import eval

def selectTrainData(tweets, targets):
    inv_topics = {v: k for k, v in tokenize_tweets.TOPICS_LONG.items()}
    inlist = []
    outcnt = 0
    for i, tweet in enumerate(tweets):
        target_keywords = tokenize_tweets.KEYWORDS.get(inv_topics.get(targets[i]))
        target_in_tweet = 0
        for key in target_keywords:
            if key.lower() in tweet.lower():
                target_in_tweet = 1
                break
        if target_in_tweet == 1:
            inlist.append(i)
        else:
            outcnt += 1
    print("Incnt", len(inlist), "Outcnt", outcnt)
    return inlist


def printInOutFiles(inlist, infile, outfileIn, outfileOut):
    outfIn = open(outfileIn, 'w')
    outfOut = open(outfileOut, 'w')
    cntr = 0
    for line in io.open(infile, encoding='windows-1252', mode='r'):  # for the Trump file it's utf-8
        if line.startswith('ID\t'):
            outfIn.write(line)
            outfOut.write(line)
        else:
            if cntr in inlist:
                outfIn.write(line)
            else:
                outfOut.write(line)
            cntr += 1

    outfIn.close()
    outfOut.close()


if __name__ == '__main__':
    tweets_gold, targets_gold, labels_gold = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)
    tweets_res, targets_res, labels_res = readTweetsOfficial("out_hillary_auto_false_targetInTweet.txt", 'windows-1252', 2)

    inlist = selectTrainData(tweets_gold, targets_gold)
    printInOutFiles(inlist, "out_hillary_auto_false_targetInTweet.txt", "out_hillary_inTwe.txt", "out_hillary_outTwe.txt")
    printInOutFiles(inlist, tokenize_tweets.FILEDEV, "_gold_hillary_inTwe.txt", "_gold_hillary_outTwe.txt")

    print("Inlist")
    eval("_gold_hillary_inTwe.txt", "out_hillary_inTwe.txt")

    print("Outlist")
    eval("_gold_hillary_outTwe.txt", "out_hillary_outTwe.txt")