# stance-semeval2016

Resources related to the Sheffield NLP group submission to the SemEval 2016 Stance Detection task.

Tokenised tweets are stored in files for quicker access, they can generated with find_tokens.py or downloaded from https://www.dropbox.com/sh/o3o2khkj4sszf2w/AAD-pWB-8p7ZJsV81ibimlrEa?dl=0
If downloaded, save them in main folder. A pre-trained autoencoder model is also saved in that folder.

Official stance data is available via https://www.dropbox.com/sh/o8789zsmpvy7bu3/AABRja7NDVPtbjSa-y3GH0jAa?dl=0  and collected unlabelled tweets are stored in https://www.dropbox.com/sh/7i2zdnet49yb1sh/AAA_AzN64JLuNlfU5pt69W8ia?dl=0

Current data sizes:

- Unlabelled tweets: 395212
- Donald Trump tweets: 16692  
- Official labelled tweets: 44389  
- Overall 25166072 tokens


bow_baseline.py runs a bow baseline, extracting 1-gram and 2-gram bow features, with end-to-end evaluation using the official eval script.

The method deep() in autoencoder.py trains the autoencoder.
After the autoencoder is trained, autoencoder_eval.py contains methods for training methods which use the autoencoder for feature extraction, also with end-to-end evaluation:

- extractFeaturesAutoencoder() extracts features by applying the autoencoder to the tweets. If the parameter "cross_features" is set to "true", features are also extracted from the targets (setting "cross_features" to "true" is currently discouraged). If the parameter is set to "added", target features are added to tweet features.
- extractFeaturesAutoencoderBOW() extract features using the autoencoder and bow

After training, model(s) can be trained with:

- train_classifiers() in bow_baseline.py for training two 2-way classifiers (on topic vs. off topic, positive vs. negative)
- train_classifier() in bow_baseline.py for training one 3-way classifier (neutral vs. pos vs. neg). If parameter "debug" is set to "true", an additional file with probabilities is printed.

The folder "output" contains output of different methods, _results.txt contains a summary of the results with explanation.