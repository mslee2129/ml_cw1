import numpy as np
import classification_helpers as ch
from classification_helpers import Node
from classification_helpers import predict_value
from classification import DecisionTreeClassifier
from evalutation_functions import accuracy
from classification_helpers import find_predominant_label
from classification_helpers import create_label_distribution_table


def random_forest (dataset, n_trees, sample_size, min_leaf, true_labels, validation_set):
    np.random.seed(42) 
    trees = []
    trees_predictions = []
    for numbers in range(n_trees):
        new_dataset = np.random.choice(dataset, replace=True, size = sample_size)     
        tree = ch.create_decision_tree(new_dataset, )
        trees.append(tree)
    
        predictions = np.zeros((validation_set.shape[0],), dtype=np.object_)
        for index in range(validation_set.shape[0]): #Going through every value we want to predict
            predictions[index] = ch.predict_value(tree, validation_set[index])
        trees_predictions.append(predictions)
   


def create_tree():
     #get sample_size samples, use np.random.choice w/ replacement
     #create an instance of the Decision Tree class 

def predict():
     #average of predictions from n_trees 
     return binary outcome 0 or 1 