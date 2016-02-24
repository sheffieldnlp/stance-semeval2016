__author__ = 'Isabelle Augenstein'

from sklearn.linear_model import LogisticRegression, SGDClassifier
import subprocess
import sys
import itertools
import io
import tokenize_tweets
from tokenize_tweets import readTweetsOfficial


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




# train one three-way classifier with SGD. Allows to specify log loss and elastic net regularisation.
def train_classifier_3waySGD(feats_train, labels_train, feats_dev, labels_dev, outfilepath, feature_vocab=[], debug='false', auto_thresh='false'):
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

    model = SGDClassifier(loss='log', penalty='l2')  # unfortunately this one doesn't have predict_proba() method debug/tuning
    model.fit(feats_train, labels)
    preds = model.predict(feats_dev)
    coef = model.coef_
    print("Label options", model.classes_)

    print("Labels", labels_dev_tr)
    print("Predictions", preds)
    print("Feat length ", feats_train[0].__len__())
    #print "Features ", feature_vocab.__len__(), "\t", feature_vocab
    #print "Weights "
    #for co in coef:
    #    print co.__len__(), "\t", co



    printPredsToFileOneModel(tokenize_tweets.FILEDEV, outfilepath, preds)




# train one three-way classifier
def train_classifier_3way(feats_train, labels_train, feats_dev, labels_dev, outfilepath, feature_vocab=[], debug='false', auto_thresh='false', useDev=True, postprocess=True):
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
        if lab == 'NONE' or lab == 'UNKNOWN':
            labels_dev_tr.append(-1)
        elif lab == 'FAVOR':
            labels_dev_tr.append(1)
        elif lab == 'AGAINST':
            labels_dev_tr.append(0)


    print("Training classifier...")

    model = LogisticRegression(penalty='l2')#, class_weight='balanced') #svm.SVC(class_weight={1: weight})
    model.fit(feats_train, labels)
    preds = model.predict(feats_dev)
    preds_prob = model.predict_proba(feats_dev)
    coef = model.coef_
    print("Label options", model.classes_)

    print("Labels", labels_dev_tr)
    print("Predictions", preds)
    print("Predictions prob", preds_prob)
    print("Feat length ", feats_train[0].__len__())
    #print "Features ", feature_vocab.__len__(), "\t", feature_vocab
    #print "Weights "
    #for co in coef:
    #    print co.__len__(), "\t", co

    if useDev == False:
        tweets_test_file = tokenize_tweets.FILEDEV
        target_short = "clinton"
    else:
        tweets_test_file = tokenize_tweets.FILETEST
        target_short = "trump"

    if auto_thresh == "true":
        print("Number dev samples:\t", len(labels_dev_tr))
        optlabels = optimiseThresh(labels_dev_tr, preds_prob, len(labels_dev_tr)/2)
        printPredsToFileOneModel(tweets_test_file, outfilepath, optlabels, len(labels_dev_tr)/2)
    else:
        printPredsToFileOneModel(tweets_test_file, outfilepath, preds)


    if postprocess == True:
        tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tweets_test_file, 'windows-1252', 2)
        targetInTweet = {}#istargetInTweet(tweets_dev, targets_dev)
        for i, tweet in enumerate(tweets_dev):
            target_keywords = tokenize_tweets.KEYWORDS.get(target_short)
            target_in_tweet = False
            for key in target_keywords:
                if key.lower() in tweet.lower():
                    target_in_tweet = True
                    break
            targetInTweet[i] = target_in_tweet

        predictions_new = []
        for i, pred_prob in enumerate(preds_prob):
            inTwe = targetInTweet[i]
            if inTwe == True:  # NONE/AGAINST/FAVOUR
                pred = 0
                if pred_prob[2] > pred_prob[1]:
                    pred = 1
                predictions_new.append(pred)
            else:
                plist = pred_prob.tolist()
                pred = plist.index(max(plist))-1
                predictions_new.append(pred)
        printPredsToFileOneModel(tweets_test_file, outfilepath, predictions_new)



    if debug == "true":

        tweets_dev, targets_dev, labels_dev = readTweetsOfficial(tokenize_tweets.FILEDEV, 'windows-1252', 2)

    #    printProbsToFileOneModel(tokenize_tweets.FILEDEV, outfilepath.replace(".txt", ".debug.txt"), preds_prob, preds)
        print("\nFeature analysis\nFeature\tNone\tAgainst\tFavor")
        for i, feat in enumerate(feature_vocab):
            print(feat, "\t", coef[0][i], "\t", coef[1][i], "\t", coef[2][i])

        print("\nActive features on dev (Hillary Clinton) per instance, coef for None/Against/Favour")
        for i, featvect in enumerate(feats_dev):
            featprint = []
            for ii, feat in enumerate(featvect):
                featname = feature_vocab[ii]
                if feat == 1.0:
                    featprint.append("[" + featname + " " + str(coef[0][ii]) + " / " + str(coef[1][ii]) + " / " + str(coef[2][ii]) + "]")
            #print labels_dev[i], "\t", tweets_dev[i], "\tFeatures:\t" , "\t".join(featprint)



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



# print predictions to file in SemEval format so the official eval script can be applied
def printPredsToFileOneModel(infile, outfile, res, skip=0):
    outf = open(outfile, 'w')
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



def getRange(x, y, stepsize):
    ret = []
    while x <= y:
        x += stepsize
        ret.append(x)
    return ret

# optimise threshold for classes on dev set for highest F1.
def optimiseThresh(labels_dev, preds_prob, howmany):

    print("Optimising threshold")

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
                print("\n--------\nConfusion matrix for dev 1 and thresh ", best_thres, "\n--------\n\t\t\t\t\t  Predicted label\n\t\t\t\t\tNon\tagainst\tfavour")
                print("True label\tNon\t", n_as_n, "     ", n_as_a, "     ", n_as_f)
                print("True label\tAgainst\t", a_as_n, "     ", a_tp, "     ", a_as_f)
                print("True label\tFavour\t\t", f_as_n, "     ", f_as_a, "     ", f_tp, "\n--------\n")


    print("Best thresh", best_thres)
    print("Best F1 on dev 1", best_f1)

    print("\nResults on dev 2 without threshold tuning")

    retlabels, for_p, for_r, for_f1, against_p, against_r, against_f1, macro_f1, a_all, a_tp, a_as_f, a_as_n, f_all, f_tp, f_as_a, f_as_n, n_as_n, n_as_f, n_as_a = computeF1ForThresh(labels_dev[howmany:], preds_prob[howmany:], [0.0, 0.0, 0.0])
    print("F1 on dev 2", for_f1, against_f1, macro_f1)

    print("\n--------\nConfusion matrix for dev 2 without threshold tuning\n--------\n\t\t\t\t\t  Predicted label\n\t\t\t\t\tNon\tagainst\tfavour")
    print("True label\tNon\t", n_as_n, "     ", n_as_a, "     ", n_as_f)
    print("True label\tAgainst\t", a_as_n, "     ", a_tp, "     ", a_as_f)
    print("True label\tFavour\t\t", f_as_n, "     ", f_as_a, "     ", f_tp, "\n--------\n")


    print("\nApplying final threshold")

    retlabels, for_p, for_r, for_f1, against_p, against_r, against_f1, macro_f1, a_all, a_tp, a_as_f, a_as_n, f_all, f_tp, f_as_a, f_as_n, n_as_n, n_as_f, n_as_a = computeF1ForThresh(labels_dev[howmany:], preds_prob[howmany:], best_thres)
    print("F1 on dev 2", for_f1, against_f1, macro_f1)

    print("\n--------\nConfusion matrix for dev 2 with best thresh\n--------\n\t\t\t\t\t  Predicted label\n\t\t\t\t\tNon\tagainst\tfavour")
    print("True label\tNon\t", n_as_n, "     ", n_as_a, "     ", n_as_f)
    print("True label\tAgainst\t", a_as_n, "     ", a_tp, "     ", a_as_f)
    print("True label\tFavour\t\t", f_as_n, "     ", f_as_a, "     ", f_tp, "\n--------")

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
    print(thresh, "\t", macro_f1)
    return retlabels, for_p, for_r, for_f1, against_p, against_r, against_f1, macro_f1, a_all, a_tp, a_as_f, a_as_n, f_all, f_tp, f_as_a, f_as_n, n_as_n, n_as_f, n_as_a



# evaluate using the original script, needs to be in same format as train/dev data
def eval(file_gold, file_pred):
    pipe = subprocess.Popen(["perl", "eval.pl", file_gold, file_pred], stdout=sys.stdout) #stdout=subprocess.PIPE)
    pipe.communicate()

