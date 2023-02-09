##############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the train_and_predict() function. 
#             You are free to add any other methods as needed. 
##############################################################################

import numpy as np
import classification_helpers as ch
from classification import DecisionTreeClassifier
from evalutation_functions import print_all_evaluation_metrics
from evalutation_functions import accuracy
from pruning import prune


def train_and_predict(x_train, y_train, x_test, x_val, y_val):
    """ Interface to train and test the new/improved decision tree.
    
    This function is an interface for training and testing the new/improved
    decision tree classifier. 

    x_train and y_train should be used to train your classifier, while 
    x_test should be used to test your classifier. 
    x_val and y_val may optionally be used as the validation dataset. 
    You can just ignore x_val and y_val if you do not need a validation dataset.

    Args:
    x_train (numpy.ndarray): Training instances, numpy array of shape (N, K) 
                       N is the number of instances
                       K is the number of attributes
    y_train (numpy.ndarray): Class labels, numpy array of shape (N, )
                       Each element in y is a str 
    x_test (numpy.ndarray): Test instances, numpy array of shape (M, K) 
                            M is the number of test instances
                            K is the number of attributes
    x_val (numpy.ndarray): Validation instances, numpy array of shape (L, K) 
                       L is the number of validation instances
                       K is the number of attributes
    y_val (numpy.ndarray): Class labels of validation set, numpy array of shape (L, )
    
    Returns:
    numpy.ndarray: A numpy array of shape (M, ) containing the predicted class label for each instance in x_test
    """

    #######################################################################
    #                 ** TASK 4.1: COMPLETE THIS FUNCTION **
    #######################################################################

    # Make sure that x and y have the same number of instances
    assert x_train.shape[0] == len(y_train), \
        "Training failed. x_train and y_train must have the same number of instances."
    assert x_val.shape[0] == len(y_val), \
        "Training failed. x_val and y_val must have the same number of instances."

    y_train_int_labels = ch.get_int_labels(y_train)
    data = ch.concat_data_helper(x_train, y_train_int_labels)

    # Creating multiple decision trees to find optimal depth
    decision_trees = []
    decision_trees.append(ch.create_decision_tree(data))
    #for max_depth in range(12,13): #choose the value here
        # I ran it until max_depth = 25. 12 was still the best
        #decision_trees.append(ch.create_decision_tree(data, max_depth))

    # Compute the accuracy of each decision tree to find out which is the best
    accuracies = []
    for tree in decision_trees:
        predictions = np.zeros((x_val.shape[0],), dtype=np.object_)    
        for index in range(x_val.shape[0]): #Going through every value we want to predict
            predictions[index] = ch.predict_value(tree, x_val[index])

        # Find the accuracy of the prediction of this specific tree
        accuracies.append(accuracy(y_val, predictions))

    # Find the maximum accuracy
    max_accuracy = max(accuracies)

    # Find the best tree based on the best accuracy
    best_tree_index = accuracies.index(max_accuracy)

    # Find the predictions of the tree with best depth, and print the evaluation metrics for it
    predictions = np.zeros((x_val.shape[0],), dtype=np.object_)    
    for index in range(x_val.shape[0]): #Going through every value we want to predict
        predictions[index] = ch.predict_value(decision_trees[best_tree_index], x_val[index])
    
    print_all_evaluation_metrics(y_val, predictions)
    
    
    # Now, Prune the best_tree
    #pruned_tree, pr_accuracy = prune(decision_trees[best_tree_index], max_accuracy, y_val, x_val)
    
    # Pruning the best tree
    pruned_tree = prune(decision_trees[best_tree_index], max_accuracy, y_val, x_val)
    
    # Finding the predictions of the best tree, and printing its evaluation metricx
    pruned_predictions = np.zeros((x_val.shape[0],), dtype=np.object_)    
    for index in range(x_val.shape[0]): #Going through every value we want to predict
        pruned_predictions[index] = ch.predict_value(pruned_tree, x_val[index])
    
    print_all_evaluation_metrics(y_val,pruned_predictions)



    # TODO: Train new classifier

    # set up an empty (M, ) numpy array to store the predicted labels 
    # feel free to change this if needed
    #predictions = np.zeros((x_test.shape[0],), dtype=np.object_)
        
    # TODO: Make predictions on x_test using new classifier        
        
    # remember to change this if you rename the variable
    #return predictions


