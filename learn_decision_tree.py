"""Implements a decision tree learner"""
# n is the number of features per instance
# m is the number of training instances

import arff  # ARFF module

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
        instances_ = instances
        norminalities_ = norminalities
        value_order_ = value_order

        m_ = len(instances)
        n_ = len(norminalities) - 1

    def make_subtree(self, instances):
        """Return a decision sub-tree
        """

        split_criteria = determine_split_candidates(instances)
    
    def determine_split_candidates(self, instances):
        """Return a list of split candidates.
        
        Each split candidate represents a decision in a DT node, and is a
        length-k list of functions, where k is the number of branches in the
        node. Each function takes as input an instance and returns a boolean.
        """

        # Iterate over all features
        for fi in range(n_):
            norminal = norminalities[fi]
            if norminal:
                pass

# Name of ARFF file
filename = "../data/heart_train.arff"

# Load ARFF file
data, metadata = arff.loadarff(filename)

# Length n+1 list of booleans for whether each feature is norminal
# Feature is numeric if it's not norminal
# The additional 1 is the class feature type
norminalities = [type_ == 'nominal' for type_ in metadata.types()]

# Get a length m list, each element is of length n+1 (features + label)
instances = data

# enumeration i is a tuple of all possible values of feature i
value_enumerations = []
for name in metadata.names():
    norminality, value_enumeration = metadata[name]
    value_enumerations.append(value_enumeration)
