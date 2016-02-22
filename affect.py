#!/usr/bin/env python

__author__ = 'Isabelle Augenstein'

import numpy as np


def getAffect(tweets):

    # Impact of gaz features, largely unspectacular
    # affect_anger.lst  -
    # affect_bad.lst o
    # affect_disgust.lst -
    # affect_fear.lst +
    # affect_joy.lst -
    # affect_sadness.lst o
    # affect_surprise.lst o
    # swear_bad.lst o      <- this is not part of WN affect, I think, something Diana created

    files = ["affect_anger.lst", "affect_bad.lst", "affect_disgust.lst", "affect_fear.lst", "affect_joy.lst", "affect_sadness.lst",
             "affect_surprise.lst"]
    vocab = ["anger", "bad", "disgust", "fear", "joy", "sadness", "surprise"]
    vects = []
    gaz = []

    for f in files:
        print(f)
        ga = []
        for line in open("wn_affect/" + f, 'r'):
            ga.append(line.split("&")[0])
        gaz.append(ga)

    for tweet in tweets:
        vect = np.zeros(len(vocab))
        for i, g in enumerate(gaz):
            affect = 0
            for entry in g:
                if tweet.__contains__(entry):
                    affect = 1  # small training set, probably doesn't make sense to introduce counts for that
                    break
            vect[i] = affect
        vects.append(vect)

    return vects, vocab