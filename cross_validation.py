import classification_helpers as ch
import numpy as np
from classification import DecisionTreeClassifier
from evalutation_functions import accuracy, print_all_evaluation_metrics
from numpy.random import default_rng
from scipy.stats import mode

def k_fold_split(n_splits, n_instances, random_generator=default_rng()):
    """ Split n_instances into n mutually exclusive splits at random.
    
    Args:
        n_splits (int): Number of splits
        n_instances (int): Number of instances to split
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list (length n_splits). Each element in the list should contain a 
            numpy array giving the indices of the instances in that split.
    """

    # generate a random permutation of indices from 0 to n_instances
    shuffled_indices = random_generator.permutation(n_instances)

    # split shuffled indices into almost equal sized splits
    split_indices = np.array_split(shuffled_indices, n_splits)

    return split_indices

def train_test_k_fold(n_folds, n_instances, random_generator=default_rng()):
    """ Generate train and test indices at each fold.
    
    Args:
        n_folds (int): Number of folds
        n_instances (int): Total number of instances
        random_generator (np.random.Generator): A random generator

    Returns:
        list: a list of length n_folds. Each element in the list is a list (or tuple) 
            with two elements: a numpy array containing the train indices, and another 
            numpy array containing the test indices.
    """

    # split the dataset into k splits
    split_indices = k_fold_split(n_folds, n_instances, random_generator)

    folds = []
    for k in range(n_folds):
        # pick k as test
        test_indices = split_indices[k]

        # combine remaining splits as train
        # this solution is fancy and worked for me
        # feel free to use a more verbose solution that's more readable
        train_indices = np.hstack(split_indices[:k] + split_indices[k+1:])

        folds.append([train_indices, test_indices])

    return folds


# Q3.2 Implementation of 10-fold Cross validation

(x, y, classes) = ch.read_dataset("data/train_full.txt")
n_folds = 10
accuracies = np.zeros((n_folds, ))
decision_trees = []
for i, (train_indices, test_indices) in enumerate(train_test_k_fold(n_folds, len(x))):
    # get the dataset from the correct splits
    x_train = x[train_indices, :]
    y_train = y[train_indices]
    x_test = x[test_indices, :]
    y_test = y[test_indices]

    # Train the Decision Trees
    decision_tree_classifier = DecisionTreeClassifier()
    decision_tree_classifier.fit(x_train, y_train)
    predictions = decision_tree_classifier.predict(x_test)
    acc = accuracy(y_test, predictions)
    accuracies[i] = acc
    # append decision tree to list
    decision_trees.append(decision_tree_classifier)

print(accuracies)
print(accuracies.mean())
print(accuracies.std())


# Q3.3 Implementation
(x_test, y_test, classes_test) = ch.read_dataset("data/test.txt")

prediction_list = []
# add predictions of each tree to list
for i in range(n_folds):
    prediction_list.append(decision_trees[i].predict(x_test))

# print('Python array: prediction_list: ', prediction_list[:][:3])
predictions = np.column_stack(tuple(prediction_list)) # now each column contains a different model's prediction for a row
# print('Numpy array: ',predictions)

# we can now take the most frequently occuring figure in each row to be our prediction
# in the case of a tie we take the value occuring nearest to the end of the array
modes, _ = mode(predictions, axis=1)

# Extract the most commonly occurring value in each row
final_predictions = modes
# print(final_predictions)

# print(decision_trees[i].predict(x_test).shape)
# print(final_predictions.shape)

acc_results = accuracy(y_test, np.squeeze(final_predictions))
# print(acc_results)