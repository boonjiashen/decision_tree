#!/usr/bin/python
"""Print prediction results of all instances of a test set
"""
import arff  # ARFF module
import optparse
import math
import DT_classifier

if __name__ == "__main__":

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
    random.seed(1)

    # Number of samples taken for each sample size
    n_samples_per_sample_size = 10

    # Percentage of training set used for training
    sample_percentages = [5, 10]

    for sample_percentage in sample_percentages:

        # Size of sample given the sample percentage
        sample_size = int(sample_percentage / 100. * len(data_list))

        # Accuracies of tree for this given sample percentage
        accuracies = []
        for ki in range(n_samples_per_sample_size):

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

        print 'For %.0f%% of training set' % (sample_percentage)
        print '\taverage accurancy is %.1f%%' % (100 * ave_accuracy)
        print '\tmin is %.1f%%' % (100 * min_accuracy)
        print '\tmax is %.1f%%' % (100 * max_accuracy)
