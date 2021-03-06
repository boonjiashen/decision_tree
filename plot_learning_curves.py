"""Implements a decision tree learner"""
# n is the number of features per instance
# m is the number of training instances

import arff  # ARFF module
#import argparse
import optparse
import math
import DT_classifier
import matplotlib.pyplot as plt

# Parse arguments
parser = optparse.OptionParser()
options, args = parser.parse_args()
assert len(args) == 3

# First positional argument: name of ARFF file
# Second positional argument: number of minimum instances to allow a node to
# split
train_filename, test_filename, min_instances = args
min_instances = int(min_instances)  # Cast to integer

#################### Declare inputs for learning #################### 

# Load ARFF file
data, metadata = arff.loadarff(train_filename)

# Change data to Python native list of lists
#data_list = [[x for x in list_] for list_ in data]
data_list = [list_ for list_ in data]

# Length n+1 list of booleans for whether each feature is norminal
# Feature is numeric if it's not norminal
# The additional 1 is the class feature type
norminalities = [type_ == 'nominal' for type_ in metadata.types()]

# enumeration i is a tuple of all possible values of feature i
value_enumerations = []
for name in metadata.names():
    norminality, value_enumeration = metadata[name]
    value_enumerations.append(value_enumeration)

#################### Plot learning curves #################### 

import random
#random.seed(1)

# Number of samples taken for each sample size
n_samples_per_sample_size = 10

# Percentage of training set used for training
sample_percentages = [5, 10, 20, 50, 100]

for sample_percentage in sample_percentages:

    # Size of sample given the sample percentage
    sample_size = int(sample_percentage / 100. * len(data_list))

    # Accuracies of tree for this given sample percentage
    accuracies = []
    for ki in (range(n_samples_per_sample_size) if sample_percentage != 100
            else [1]):

        # Get a length m list, each element is of length n+1 (features + label)
        random.shuffle(data_list)
        sample = data_list[:sample_size]

        # Instantiate tree learner
        classifier = DT_classifier.DT_classifier(sample, norminalities, value_enumerations,
                min_instances)

        # Fit classifer
        classifier.fit()

        # Get accuracy of this fit
        testset, metadata = arff.loadarff(test_filename)
        testset = [list_ for list_ in testset]
        n_correct = sum(
                [classifier.predict(instance) == instance[-1]
                for instance in testset])
        accuracy = 1. * n_correct / len(testset)

        # Update list of accuracies for this point on learning curve
        accuracies.append(accuracy)

    # Calculate metrices for accuracy
    ave_accuracy = sum(accuracies) / len(accuracies)
    min_accuracy = min(accuracies)
    max_accuracy = max(accuracies)

    # Collect metrices in their respective lists
    try:
        ave_accuracies, min_accuracies, max_accuracies
    except NameError:
        ave_accuracies, min_accuracies, max_accuracies =  \
                [], [], []
    ave_accuracies.append(ave_accuracy)
    min_accuracies.append(min_accuracy)
    max_accuracies.append(max_accuracy)

    print 'For %.0f%% of training set' % (sample_percentage)
    print '\taverage accurancy is %.1f%%' % (100 * ave_accuracy)
    print '\tmin is %.1f%%' % (100 * min_accuracy)
    print '\tmax is %.1f%%' % (100 * max_accuracy)

############# Plot learning curves biatch #################### 

plt.figure()

# Draw plots
plt.plot(sample_percentages, max_accuracies, 'g', label="maximum accuracy")
plt.plot(sample_percentages, max_accuracies, 'go')
plt.plot(sample_percentages, ave_accuracies, 'b', label="average accuracy", lw=2.)
plt.plot(sample_percentages, ave_accuracies, 'bo', lw=2.)
plt.plot(sample_percentages, min_accuracies, 'r', label="minimum accuracy")
plt.plot(sample_percentages, min_accuracies, 'ro')

# Add text to plot
plt.legend(loc="best")
title = "Accuracy versus training set size\n" +  \
        "training data: " + train_filename +  "\n" +  \
        "testing data: " + test_filename
plt.title(title)
plt.xlabel("Percentage of instances used in training set")
plt.ylabel("Accuracy")

# Change limits of plot
# y axis limits are in multiples of y_granularity
x1, x2, y1, y2 = plt.axis()
y_granularity = .2
y1 = math.floor(y1 / y_granularity) * y_granularity
y2 = math.ceil(y2 / y_granularity) * y_granularity
plt.axis((0, x2, y1, y2))

plt.show()
