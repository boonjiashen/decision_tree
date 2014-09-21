"""Implements a decision tree learner"""
# n is the number of features per instance
# m is the number of training instances

import arff  # ARFF module
#import argparse
import optparse
import math

def get_entropy(items):
    "Calculate the entropy of a list of items"

    item_set = set(items)  # Get list of unique items

    entropy = 0
    for unique_item in item_set:
        probability = 1. * items.count(unique_item) / len(items)
        entropy = entropy - probability * math.log(probability, 2)

    return entropy

class DT_learner():
    """Decision tree learner for a binary class.
    """

    instances_ = None
    norminalities_ = None
    value_enumeration_ = None
    tree_ = None  # Decision tree
    m = 0  # Number of training instances
    n = 0  # Number of features
    min_instances = 0  # Min no. of instances at a node that allows splits
    priority_class = None  # class that wins in a tie-breaker

    def __init__(self, instances, norminalities, value_enumerations):
        """Constructor for decision tree learner

        instances - 
            a m x n+1 list of lists
            The last column is the class of the instances
        norminalities -
            n+1 length list of booleans.
            norminalities[i] answers whether feature i is norminal (numeric
            otherwise)
        value_enumerations -
            n+1 length list of tuples.
            If norminalities[i] is True, value_enumeration[i] is a tuple of the
            possible norminal values of feature[i].
        """
        self.instances = instances
        self.norminalities = norminalities
        self.value_enumerations = value_enumerations

        self.m = len(instances)
        self.n = len(instances[0]) - 1

        assert len(self.value_enumerations) == self.n + 1
        assert len(self.norminalities) == self.n + 1

    def set_priority_class(self, instances):
        "Establish class priority: first in list has highest priority"

        assert len(instances) > 0
        self.priority_class = instances[0][-1]

    def get_conditional_entropy(self, instances, split_criterion):
        """Return the entropy of a set of instances conditioned on
        a split criterion.

        See determine_split_candidates() for the specification for a split
        criterion.
        """

        # Partition instances into their rightful branches down the node
        partitions = []
        for instance in instances:
            partition_index = self.look_up_branch_index(
                    instance, split_criterion)

            # Expand number of partitions as necessary
            while len(partitions) < partition_index + 1:
                partitions.append([])

            # Add instance to its rightful partition
            partitions[partition_index].append(instance)

        # Calculate conditional entropy
        cond_entropy = 0
        for partition in partitions:

            # Probability of an instance being in this partition
            probability = 1. * len(partition) / len(instances)

            # Entropy of this partition
            labels = [instance[-1] for instance in partition]
            entropy = get_entropy(labels)

            cond_entropy = cond_entropy + probability * entropy

        return cond_entropy

    def look_up_branch_index(self, instance, split_criterion):
        """Return the index of a branch that an instances traverses at a node
        represented by split_criterion"""

        feature_index, threshold = split_criterion
        norminal = self.norminalities[feature_index]  # Is feature norminal?

        # Value of feature of this instance
        feature_value = instance[feature_index]

        if norminal:
            # Branches of norminal features are ordered based on
            # value_enumerations
            # e.g. if the enumeration is ['red', 'green', blue'] and the value
            # is 'green', the branch index is 1
            branch_index =  \
                self.value_enumerations[feature_index].index(feature_value)
        else:
            # For numeric features:
            #The left branch of such a split should represent values that are
            #less than or equal to the threshold.
            if feature_value <= threshold:
                branch_index = 0
            else:
                branch_index = 1

        return branch_index


    def make_subtree(self, instances):
        """Return a decision sub-tree
        """

        split_criteria = self.determine_split_candidates(instances)

        # Stop criterion 1: all classes are same
        assert len(instances) > 0
        all_classes_same = all([instance[-1] == instances[0][-1]
            for instance in instances])

        # Stop criterion 2: less than some min no. of instances
        few_instances = len(instances) < self.min_instances

        # Stop criterion 3: no feature has positive information gain
        none_have_info_gain = False  # TODO implement this

        # Stop criterion 4: no more features to split on
        no_remaining_features = len(features_remaining) == 0
    
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
        for fi in range(self.n):
            if fi not in features_remaining:
                continue

            # Get candidates according to norminality of this feature
            norminal = self.norminalities[fi]
            if norminal:
                # Norminal features only have one candidate
                curr_candidates = [(fi, None)]
            else:
                # Numerical features may have more than one candidate
                curr_candidates = determine_candidate_numeric_splits(
                        instances, fi)

            split_candidates.extend(curr_candidates)

    def determine_candidate_numeric_splits(self, instances, feature_ind):
        """Return a list of split candidates.

        See determine_split_candidates() for representation
        """
        # TODO implement the full version of this function

        # Check that feature is numeric
        assert not self.norminalities[feature_ind]

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
random.seed(1)
random.shuffle(classifier.instances)
subset = classifier.instances[:5]

print 'subset is'
for instance in subset: print instance
classifier.get_conditional_entropy(subset, (1, None))
