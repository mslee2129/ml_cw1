##############################################################################
# Introduction to Machine Learning
# Coursework 1 example execution code
# Prepared by: Josiah Wang
##############################################################################

import numpy as np

import classification as cl # ADDED BY US

# from classification import DecisionTreeClassifier # COMMENTED BY US
from improvement import train_and_predict

if __name__ == "__main__":

    # ADDED BY US
    # Reading and examining the datasets
    (x, y, classes) = cl.read_dataset("data/train_full.txt")
    cl.examine_dataset(x,y,classes, "train_full")

    #(x, y, classes) = cl.read_dataset("data/train_sub.txt")
    # cl.examine_dataset(x,y,classes, "train_sub")

    (x, y, classes) = cl.read_dataset("data/train_noisy.txt")
    cl.examine_dataset(x,y,classes, "train_noisy")




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
    
    x_val = np.array([
                [6,7,2],
                [3,1,3]
            ])
    y_val = np.array(["A", "C"])
                   
    print("Training the improved decision tree, and making predictions on the test set...")
    predictions = train_and_predict(x, y, x_test, x_val, y_val)
    print("Predictions: {}".format(predictions))
    
    """