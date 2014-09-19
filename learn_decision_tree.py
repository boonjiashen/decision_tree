"""Implements a decision tree learner"""
# n is the number of features per instance
# m is the number of training instances

import arff  # ARFF module
#import argparse
import optparse

class DT_learner():

    instances_ = None
    norminalities_ = None
    value_order_ = None
    tree_ = None  # Decision tree
    m_ = 0  # Number of training instances
    n_ = 0  # Number of features
    min_instances = 0  # Min no. of instances at a node that allows splits

    def __init__(self, instances, norminalities, value_order):
        """Constructor for decision tree learner

        instances - a m x n+1 list of lists
        norminalities -
            n+1 length list of booleans.
            norminalities[i] answers whether feature i is norminal (numeric
            otherwise)
        value_order -
            n+1 length list of tuples.
            If norminalities[i] is True, value_order[i] is a tuple of the
            possible norminal values of feature[i].
        """
        self.instances_ = instances
        self.norminalities_ = norminalities
        self.value_order_ = value_order

        self.m_ = len(instances)
        self.n_ = len(norminalities) - 1

    def make_subtree(self, instances):
        """Return a decision sub-tree
        """

        split_criteria = determine_split_candidates(instances)
    
    def determine_split_candidates(self, instances, features_remaining):
        """Return a list of split candidates.
        
        Each split candidate represents a decision in a DT node, and is a
        2-ple.
        The first element is the index of the feature (between 0 and n_-1
        inclusive). If the feature is nominal, the second element is None,
        otherwise the second element is the threshold of the numeric feature.
        """

        # Iterate over all features
        split_candidates = []
        for fi in range(self.n_):
            if fi not in features_remaining:
                continue
            norminal = self.norminalities[fi]
            if norminal:
                split_candidate = (fi, None)
            else:
                #TODO implement this
                pass

    def determine_candidate_numeric_splits(self, instances, feature_ind):
        """Return a list of split candidates.

        See determine_split_candidates() for representation
        """
        # TODO implement the full version of this function

        # Check that feature is numeric
        assert not self.norminalities_[feature_ind]

        # Sort instances by feature
        instances.sort(key=lambda x: x[feature_ind])

        # Look for mid-point where instances[i] and
        # instances[i+1] have different classes
        candidates = []
        for i in range(len(instances) - 1):
            if instances[i][-1] != instances[i+1][-1]:
                midpoint = (
                        instances[i][feature_ind] +
                        instances[i+1][feature_ind]) / 2
                candidates.append((feature_ind, midpoint, ))

        return candidates

# Parse arguments
parser = optparse.OptionParser()
options, args = parser.parse_args()
assert len(args) == 1

# Name of ARFF file is the first positional argument
filename = args[0]

# Load ARFF file
data, metadata = arff.loadarff(filename)

# Change data to Python native list of lists
#data_list = [[x for x in list_] for list_ in data]
data_list = [list_ for list_ in data]

# Length n+1 list of booleans for whether each feature is norminal
# Feature is numeric if it's not norminal
# The additional 1 is the class feature type
norminalities = [type_ == 'nominal' for type_ in metadata.types()]

# Get a length m list, each element is of length n+1 (features + label)
instances = data_list

# enumeration i is a tuple of all possible values of feature i
value_enumerations = []
for name in metadata.names():
    norminality, value_enumeration = metadata[name]
    value_enumerations.append(value_enumeration)

# Instantiate tree learner
classifier = DT_learner(instances, norminalities, value_enumerations)
import random
random.shuffle(classifier.instances_)
subset = classifier.instances_[:10]
