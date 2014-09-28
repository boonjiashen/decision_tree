import math
import DT_classifier

# Create dummy classifier
training_set = [[0., "dog"], ]
norminalities = [False, True]
value_enumerations = [None, True]
min_instances = 5
classifier = DT_classifier.DT_classifier(training_set, norminalities,
        value_enumerations, min_instances)

# For the following tests:
# Instance is described by [feature, label]
#---1---0---  <- number line

# Two different labels at different instances
#---1---0---  <- number line
# We should get the mid-point of the two values
candidates = classifier.determine_candidate_numeric_splits(
        [[0, 1], [1, 0]], 0)
assert len(candidates) == 1
feature_ind, threshold = candidates[0]
assert threshold == 0.5

# Two same labels at different feature values
#---1---1---  <- number line
# We should get nothing returned
candidates = classifier.determine_candidate_numeric_splits(
        [[0, 1], [1, 1]], 0)
assert len(candidates) == 0

#---0---0---  <- number line
#       1
# Expect a split at midpoint
candidates = classifier.determine_candidate_numeric_splits(
        [[0, 0], [1, 0], [1, 1]], 0)
assert len(candidates) == 1
feature_ind, threshold = candidates[0]
assert threshold == 0.5

#---0---0---1  <- number line
#       1   0
# Expect a split at both midpoints
candidates = classifier.determine_candidate_numeric_splits(
        [[0, 0], [1, 0], [1, 1], [2, 1], [2, 0]], 0)
assert len(candidates) == 2
assert candidates[0][1] == 0.5
assert candidates[1][1] == 1.5
