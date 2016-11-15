# -*- mode: Python; coding: utf-8 -*-

from classifier import Classifier
import numpy, math, sys, scipy
from random import shuffle
from nltk.corpus import stopwords
import matplotlib.pyplot as plt

class MaxEnt(Classifier):
    def __init__(self):
        model = {}
        features = {}
        labels = {}

    def save_features(self, file):
        """Save the current model to the given file."""
        if isinstance(file, basestring):
            with open(file, "wb") as file:
                self.save(file)
        else:
            dump(self.features, file, HIGHEST_PICKLE_PROTOCOL)

    def save_labels(self, file):
        """Save the current model to the given file."""
        if isinstance(file, basestring):
            with open(file, "wb") as file:
                self.save(file)
        else:
            dump(self.labels, file, HIGHEST_PICKLE_PROTOCOL)

    def load_features(self, file):
        """Load a saved model from the given file."""
        if isinstance(file, basestring):
            with open(file, "rb") as file:
                self.load(file)
        else:
            self.features = load(file)

    def load_labels(self, file):
        """Load a saved model from the given file."""
        if isinstance(file, basestring):
            with open(file, "rb") as file:
                self.load(file)
        else:
            self.labels = load(file)

    def lambda_weight(self, label, feature):
        return self.model.item(self.labels[label], feature)

    def create_feature_vectors(self, instance):
        feature_vector = [self.features[feature] for feature in instance.features() if feature in self.features]
        feature_vector.append(0)
        return feature_vector

    def accuracy(classifier, test, verbose=sys.stderr):
        correct = [classifier.classify(x) == x.label for x in test]
        if verbose:
            print >> verbose, "%.2d%% " % (100 * sum(correct) / len(correct)),
        return float(sum(correct)) / len(correct)

    def train(self, instances, dev_instances=None):
        """Construct a statistical model from labeled instances."""

        #create dict mapping features => ids
        features = 1
        feat_id_dict = {"___BIAS___": 0}
        labels = 0
        labels_dict = {}

        # get features
        stop = set(stopwords.words('english'))
        for instance in instances:
            try:
                x = labels_dict[instance.label]
            except KeyError:
                labels_dict[instance.label] = labels
                labels += 1
            for feature in instance.features():
                if feature not in stop:
                    try:
                        x = feat_id_dict[feature]
                    except KeyError:
                        feat_id_dict[feature] = features
                        features += 1

        # create parameters matrix
        self.model = numpy.zeros((labels, features))
        self.labels = labels_dict
        self.save_labels("labels.db")
        self.features = feat_id_dict
        self.save_features("features.db")

        # create feature vectors
        for instance in instances:
            instance.feature_vector = self.create_feature_vectors(instance)
        for instance in dev_instances:
            instance.feature_vector = self.create_feature_vectors(instance)

        #gradient descent
        self.train_sgd(instances, dev_instances, .0001, 1000)

    def train_sgd(self, train_instances, dev_instances, learning_rate, batch_size):
        """Train MaxEnt model with Mini-batch Stochastic Gradient 
        """
        negative_log_likelihood = sys.maxint
        no_change_count = 0
        while True:
            minibatches = self.chop_up(train_instances, batch_size)
            for minibatch in minibatches:
                gradient = self.compute_gradient(minibatch)
                self.model += gradient * learning_rate
            new_negative_log_likelihood = self.compute_negative_log_likelihood(dev_instances)
            if new_negative_log_likelihood >= negative_log_likelihood:
                no_change_count += 1
                print "No change ", no_change_count
                print "log likelihood was: ", new_negative_log_likelihood
                if no_change_count == 5:
                    break
            else:
                no_change_count = 0
                negative_log_likelihood = new_negative_log_likelihood
                self.save("model.db")
                print("log likelihood is: " + str(negative_log_likelihood))

    def compute_observed_counts(self, minibatch):
        observed_counts = numpy.zeros((len(self.labels), len(self.features)))
        for instance in minibatch:
            for feature in instance.feature_vector:
                observed_counts[self.labels[instance.label]][feature] += 1
        return observed_counts

    def compute_estimated_counts(self, minibatch):
        estimated_counts = numpy.zeros((len(self.labels), len(self.features)))
        for instance in minibatch:
            for label in self.labels:
                prob = self.compute_conditional_probability(label, instance)
                for feature in instance.feature_vector:
                    estimated_counts[self.labels[label]][feature] += prob
        return estimated_counts

    def compute_gradient(self, minibatch):
        obs = self.compute_observed_counts(minibatch)
        est = self.compute_estimated_counts(minibatch)
        return obs - est

    def chop_up(self, train_instances, batch_size):
        shuffle(train_instances)
        minibatches = [train_instances[x:x + batch_size] for x in range(0, len(train_instances), batch_size)]
        return minibatches

    def compute_conditional_probability(self, lbl, instance):
        top = sum([self.lambda_weight(lbl, feature) for feature in instance.feature_vector])
        bottom = [sum([self.lambda_weight(label, feature) for feature in instance.feature_vector]) for label in self.labels]
        posterior = math.exp(top - scipy.misc.logsumexp(bottom))
        return posterior

    def compute_negative_log_likelihood(self, batch):
        log_likelihood = sum([math.log(self.compute_conditional_probability(instance.label, instance)) for instance in batch])
        return log_likelihood * -1

    def classify(self, instance):
        best = ""
        prob = 0
        instance.feature_vector = self.create_feature_vectors(instance)
        for label in self.labels:
            new_prob = self.compute_conditional_probability(label, instance)
            if new_prob > prob:
                best = label
                prob = new_prob
        return best
