#############################################################################
# Introduction to Machine Learning
# Coursework 1 Skeleton code
# Prepared by: Josiah Wang
#
# Your tasks: Complete the fit() and predict() methods of DecisionTreeClassifier.
# You are free to add any other methods as needed. 
##############################################################################

import matplotlib.pyplot as plt
import numpy as np
from numpy.random import default_rng


#############################################################################
# Loading and examining the dataset
##############################################################################

def read_dataset(filepath):
    """ Read in the dataset from the specified filepath

    Args:
        filepath (str): The filepath to the dataset file

    Returns:
        tuple: returns a tuple of (x, y, classes), each being a numpy array. 
               - x is a numpy array with shape (N, K), 
                   where N is the number of instances
                   K is the number of features/attributes
               - y is a numpy array with shape (N, ), and should be integers from 0 to C-1
                   where C is the number of classes 
               - classes : a numpy array with shape (C, ), which contains the 
                   unique class labels corresponding to the integers in y
    """

    x = []
    y_labels = []
    for line in open(filepath):
        if line.strip() != "": # handle empty rows in file
            row = line.strip().split(",")
            x.append(list(map(float, row[:-1]))) 
            y_labels.append(row[-1])
    
    [classes, y] = np.unique(y_labels, return_inverse=True) 

    x = np.array(x)
    y = np.array(y)
    return (x, y, classes) #return classes?

def examine_dataset (x ,y, classes, dataset_name = ""):
    print("############", dataset_name, "############")

    print("---- Shapes ---- ")
    print("The shape of x is: ", x.shape)
    print("The shape of y is: ", y.shape)
    #print("The shape of classes is: ", classes.shape)
    print()

    print(classes, "\n")

    print("---- Min, Max, Mean ----")
    print("Minimum of x :", x.min(axis=0))
    print("Maximum of x :", x.max(axis=0))
    print("Mean of x :", x.mean(axis=0), "\n")

    print("Minimum of y:", y.min(axis=0)) ## THIS IS NOT GIVING US WHAT WE NEED
    print("Maximum of y:", y.max(axis=0)) ## THIS IS NOT GIVING US WHAT WE NEED
    print("Mean of y :", y.mean(axis=0), "\n") ## THIS IS NOT GIVING US WHAT WE NEED

    """ ------------  Number of instances  ---------------- """
    # Finding the number of instances of each class
    num_classes = classes.size
    l = [0] *  num_classes
    for i in range(len(y)):
        l[y[i]] += 1

    # Printing number of obs in each class
    print("Num observations in each class: \n")
    for i in range(num_classes):
        print("Class", i, ":", round(l[i]))
    
    # Showing a graphical representation of the repartition (#) of observations by classes
    plt.bar(range(num_classes), l, )
    title_str = "Number of observation per classes. Dataset: " + dataset_name
    plt.title(title_str)
    plt.xlabel('Class number')
    plt.ylabel('Number of observations')
    # plt.show()
    file_name = "graphs/numObs_" + dataset_name
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file

    """ ------------  RATIO  ---------------- """
    # Getting the ratio of each class
    ratio = [0] * num_classes
    print("Ratio of each class: \n")
    for i in range(num_classes):
        ratio[i] = float(l[i]/len(y)) * 100
        print("Class", i, ":", round(ratio[i]), "%")

    # Showing a graphical representation of the repartition (%) of observations by classes
    plt.bar(range(num_classes), ratio, )
    title_str = "Ratio of dataset: " + dataset_name
    plt.title(title_str)
    plt.xlabel('Class number')
    plt.ylabel('Percentage of values (weight of the class)')
    # plt.show()
    file_name = "graphs/ratio_" + dataset_name
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file


#############################################################################
# Decision Tree
##############################################################################


class DecisionTreeClassifier(object):
    """ Basic decision tree classifier
    
    Attributes:
    is_trained (bool): Keeps track of whether the classifier has been trained
    
    Methods:
    fit(x, y): Constructs a decision tree from data X and label y
    predict(x): Predicts the class label of samples X
    prune(x_val, y_val): Post-prunes the decision tree
    """

    def __init__(self):
        self.is_trained = False
    

    def fit(self, x, y):
        """ Constructs a decision tree classifier from data
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (N, K) 
                           N is the number of instances
                           K is the number of attributes
        y (numpy.ndarray): Class labels, numpy array of shape (N, )
                           Each element in y is a str 
        """
        
        # Make sure that x and y have the same number of instances
        assert x.shape[0] == len(y), \
            "Training failed. x and y must have the same number of instances."
        
        #######################################################################
        #                 ** TASK 2.1: COMPLETE THIS METHOD **
        #######################################################################    
        

        
        # set a flag so that we know that the classifier has been trained
        self.is_trained = True
        
    
    def predict(self, x):
        """ Predicts a set of samples using the trained DecisionTreeClassifier.
        
        Assumes that the DecisionTreeClassifier has already been trained.
        
        Args:
        x (numpy.ndarray): Instances, numpy array of shape (M, K) 
                           M is the number of test instances
                           K is the number of attributes
        
        Returns:
        numpy.ndarray: A numpy array of shape (M, ) containing the predicted
                       class label for each instance in x
        """
        
        # make sure that the classifier has been trained before predicting
        if not self.is_trained:
            raise Exception("DecisionTreeClassifier has not yet been trained.")
        
        # set up an empty (M, ) numpy array to store the predicted labels 
        # feel free to change this if needed
        predictions = np.zeros((x.shape[0],), dtype=np.object)
        
        
        #######################################################################
        #                 ** TASK 2.2: COMPLETE THIS METHOD **
        #######################################################################
        
    
        # remember to change this if you rename the variable
        return predictions
        
