from naivebayes import NaiveBayesClassifier
from featurizer import featurize_bigram, tokenize_whitespace, featurize_unigram, featurize_unigram
import metrics


def read_data(filename):
    with open(filename, encoding ="utf-8") as fin:
        X = []
        Y = []
        for line in fin:
            x, y = line.strip().split('\t')
            X.append(x)
            Y.append(y)
    return X, Y


def main():
    train_y, train_x = read_data("data/SH-TTC/train.tsv")
    dev_y, dev_x = read_data("data/SH-TTC/dev.tsv")


    # TODO: replace with your own featurize and tokenize functions

        #bigrams
    featurized_train_x = [featurize_bigram(tokenize_whitespace(x)) for x in train_x]
    featurized_dev_x = [featurize_bigram(tokenize_whitespace(x)) for x in dev_x]
    
        #trigrams
    # featurized_train_x = [featurize_trigram(tokenize_whitespace(x)) for x in train_x]
    # featurized_dev_x = [featurize_trigram(tokenize_whitespace(x)) for x in dev_x]

        #unigrams
    # featurized_train_x = [featurize_unigram(tokenize_whitespace(x)) for x in train_x]
    # featurized_dev_x = [featurize_unigram(tokenize_whitespace(x)) for x in dev_x]
    
    model = NaiveBayesClassifier()
    model.train(featurized_train_x, train_y)
    predictions = [model.predict(x) for x in featurized_dev_x]

    print(metrics.precision(dev_y, predictions))
    print(metrics.recall(dev_y, predictions))
    print(metrics.f1score(dev_y, predictions))


if __name__ == "__main__":
    main()
