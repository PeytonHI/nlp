from collections import Counter, defaultdict
from math import log


class NaiveBayesClassifier:
    def __init__(self):
        self.trained = False
        self.label_count = defaultdict(int) 
        self.feature_count = defaultdict(dict)
        self.total_sentences = 0

        # raise NotImplementedError()

    def train(self, X: list[dict], y: list[str]):
        """
        Learn model parameters p(w | y) and p(y)

        X: list of feature dictionaries
        y: list of labels
        """     
        self.trained = True
        # raise NotImplementedError()

        # ({"sherlock", "holmes"} , label)
        for features, label in zip(X, y):
            if label not in self.feature_count: # make new dict if label not in feature_count
                self.feature_count[label] = {}

            # nested dict: {label: {features}} add features to feature_count dict
            for feature in features:
                if feature not in self.feature_count[label]:
                    self.feature_count[label][feature] = 0
                self.feature_count[label][feature] += 1

            self.label_count[label] += 1 # update label and count in label_count dict
            self.total_sentences += 1 

        return self.label_count, self.total_sentences, self.feature_count
      
      
    def prior(self, y):
        """
        Return the prior p(y)
        """
        if not self.trained:
            raise Exception("Must train first!")
        # raise NotImplementedError()

        label_count = self.label_count[y]
        label_length = self.total_sentences
        
        prob_prior_y = log((label_count + 1) / (self.total_sentences + label_length)) # log of probabiltiy of prior y equals logprior[y] ← log(y labelled sentences + 1 for laplace smoothing, divided by total_num_sentences plus length of all labels )

        return prob_prior_y
    

    def likelihood(self, x, y):
        """
        Return the likelihood p(x | y)
        """
        if not self.trained:
            raise Exception("Must train first!")
        # raise NotImplementedError()

        feature_count = self.feature_count
        label_length = self.total_sentences

        y_labeled_values = feature_count[y].get(x, 0) # get feature counts from all y labeled sentence

        total_y_value = y_labeled_values + label_length # all y label counts from feature counts and total y labels

        likelihood = -log((y_labeled_values + 1) / total_y_value) #laplace smoothing and # of times bigram appears in a y-labeled sentence divided by total number of bigrams in all y-labeled sentences

        return likelihood
    

    def predict(self, x: dict):
        """
        Return the most likely label for the given feature dictionary
        """
        # hint: you may want to call the prior and likelihood functions here
        if not self.trained:
            raise Exception("Must train first!")
        # raise NotImplementedError() 

        max_prob = 0
        max_label = None
        # for each feature, get the probability of prior label and likelihood of each feature then set new likely label
        for label in self.label_count:
            probability_prior_label = self.prior(label)

            for feature in x.keys():
                likelihood = self.likelihood(feature, label)

                predict_value = probability_prior_label + likelihood

                if predict_value > max_prob:
                    max_prob = predict_value
                    max_label = label

        print(x.keys(), max_label)  

        return max_label
