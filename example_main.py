##############################################################################
# Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np

# import classification as cl # ADDED BY US
import classification_helpers as ch
from classification import DecisionTreeClassifier # COMMENTED BY US
from improvement import train_and_predict
from evalutation_functions import print_all_evaluation_metrics
from evalutation_functions import accuracy
from pruning import prune

if __name__ == "__main__":

    # ############# QUESTION 1 #######################
    # Reading and examining the datasets
    #(x, y, classes) = ch.read_dataset("data/tot.txt")
    #ch.examine_dataset(x,y,classes, "simple1")

    #(x, y, classes) = ch.read_dataset("data/train_sub.txt")
    # ch.examine_dataset(x,y,classes, "train_sub")

    #(x, y, classes) = ch.read_dataset("data/train_noisy.txt")
    #ch.examine_dataset(x,y,classes, "train_noisy")

    # Q1.3
    # (x_noisy, y_noisy, classes_noisy) = ch.read_dataset("data/train_noisy.txt")
    # noisy_data = ch.concat_data_helper(x_noisy, y_noisy)
    # (x, y, classes) = ch.read_dataset("data/train_full.txt")
    # clean_data = ch.concat_data_helper(x,y)
    # ch.noisy_data_comparison(clean_data, noisy_data)


    ############### QUESTION 2 ###############

    # load the test set
    (x_test, y_test, classes_test) = ch.read_dataset("data/test.txt")

    print("\nTraining model on train_full")
    full_classifier = DecisionTreeClassifier()
    (x_full, y_full, classes_full) = ch.read_dataset("data/train_full.txt")
    full_classifier.fit(x_full, y_full)
    full_predictions = full_classifier.predict(x_test)
    print_all_evaluation_metrics(y_test, full_predictions)

    print("\nTraining model on train_noisy")
    noisy_classifier = DecisionTreeClassifier()
    (x_noisy, y_noisy, classes_noisy) = ch.read_dataset("data/train_noisy.txt")
    noisy_classifier.fit(x_noisy, y_noisy)
    noisy_predictions = noisy_classifier.predict(x_test)
    print_all_evaluation_metrics(y_test, noisy_predictions)

    print("\nTraining model on train_sub")
    sub_classifier = DecisionTreeClassifier()
    (x_sub, y_sub, classes_sub) = ch.read_dataset("data/train_sub.txt")
    sub_classifier.fit(x_sub, y_sub)
    sub_predictions = sub_classifier.predict(x_test)
    print_all_evaluation_metrics(y_test, sub_predictions)

    # print("---- Welcome to Wonderland ----")
    # print("---- Loading training dataset ----")
    # x,y,c  = ch.read_dataset("./data/train_full.txt")
    # data = ch.concat_data_helper(x,y)   
    
    # print("---- Loading validation dataset ----")
    # x_validation, y_validation, c_validation  = ch.read_dataset("./data/validation.txt")
    
    # print("---- Loading testing dataset ----")
    # x_test, y_test, class_test  = ch.read_dataset("./data/test.txt")
    
    # print("---- Launching train and predict ----")
    # predicitions = train_and_predict(x,y,x_test, x_validation, y_validation)
    
    """"
    print("---- Loading dataset ----")
    x,y,c  = ch.read_dataset("./data/train_full.txt")
    print_all_evaluation_metrics(y_test, predictions)
    #tree = ch.create_decision_tree(data)
    #tree.recursive_print()
    
    print("PRUNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    pruned_classifier = DecisionTreeClassifier()
    # WE NEED TO CHANGE THIS CAUSE I AM PRUNING ON THE TEST SET
    pruned_classifier = prune(classifier, classifier.decision_tree, accuracy(y_test, predictions), y_test, x_test) 
    new_predictions = pruned_classifier.predict(x_test)
    print_all_evaluation_metrics(y_test, new_predictions)

    """
    """
    print("Loading the training dataset...");
    x = np.array([
            [5,7,1],
            [4,6,2],
            [4,6,3], 
            [1,3,1], 
            [2,1,2], 
            [5,2,6]
        ])
    
    y = np.array(["A", "A", "A", "C", "C", "C"])

    print("Training the decision tree...")
    classifier = DecisionTreeClassifier()
    classifier.fit(x, y)

    print("Loading the test set...")
    
    x_test = np.array([
                [1,6,3], 
                [0,5,5], 
                [1,5,0], 
                [2,4,2]
            ])
    
    y_test = np.array(["A", "A", "C", "C"])
    
    print("Making predictions on the test set...")
    predictions = classifier.predict(x_test)
    print("Predictions: {}".format(predictions))
    
    print_all_evaluation_metrics(y_test, predictions)

    """
    """
    x_val = np.array([
                [6,7,2],
                [3,1,3]
            ])
    y_val = np.array(["A", "C"])
                   
    print("Training the improved decision tree, and making predictions on the test set...")
    predictions = train_and_predict(x, y, x_test, x_val, y_val)
    print("Predictions: {}".format(predictions))
    """