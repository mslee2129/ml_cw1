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

if __name__ == "__main__":

    # ############# QUESTION 1 #######################
    # Reading and examining the datasets
    #(x, y, classes) = ch.read_dataset("data/simple1.txt")
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
    #x,y,c  = ch.read_dataset("./data/train_full.txt")
    #data = ch.concat_data_helper(x,y)
    #att_index, split_index = ch.find_optimal_node(data)
    #print(ch.find_optimal_node(data))
    #print(ch.make_split(data, att_index, split_index))
    
    #tree = ch.create_decision_tree(data)
    #tree.recursive_print()
    
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
    x_val = np.array([
                [6,7,2],
                [3,1,3]
            ])
    y_val = np.array(["A", "C"])
                   
    print("Training the improved decision tree, and making predictions on the test set...")
    predictions = train_and_predict(x, y, x_test, x_val, y_val)
    print("Predictions: {}".format(predictions))
    """