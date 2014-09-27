"""Plot m against accuracy of a decision tree classifier.

m is the minimum number of instances at a node during learning to allow the
node to split further."""

import arff  # ARFF module
import random
#import argparse
import optparse
import math
import DT_classifier
import matplotlib.pyplot as plt

# Parse arguments
parser = optparse.OptionParser()
options, args = parser.parse_args()
assert len(args) == 2

# First positional argument: name of ARFF file
# Second positional argument: number of minimum instances to allow a node to
# split
train_filename, test_filename = args

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

#random.seed(1)

# Parameter m - number of minimum instances
min_instance_params = [2, 5, 10, 20]
accuracies = []

for min_instances in min_instance_params:

    sample = data_list

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

    print 'For %i min instances' % (min_instances)
    print 'accuracy %f' % (accuracy)
    #classifier.print_tree(metadata.names())

############# Plot learning curves biatch #################### 

plt.figure()

# Draw plot
plt.plot(min_instance_params, accuracies, "b")
plt.plot(min_instance_params, accuracies, "bo")

# Add text to plot
title = "Accuracy versus m\n" +  \
        "training data: " + train_filename +  "\n" +  \
        "testing data: " + test_filename
plt.title(title)
plt.xlabel("m (min instances required for a node to split)")
plt.ylabel("accuracy")

# Change limits of plot
# y axis limits are in multiples of y_granularity
x1, x2, y1, y2 = plt.axis()
y_granularity = .2
y1 = math.floor(min(accuracies) / y_granularity) * y_granularity
y2 = math.ceil(max(accuracies) / y_granularity) * y_granularity
plt.axis((0, x2, y1, y2))

plt.show()
