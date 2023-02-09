import numpy as np
from classification_helpers import Node
from classification_helpers import predict_value
from classification import DecisionTreeClassifier
from evalutation_functions import accuracy
from classification_helpers import find_predominant_label
from classification_helpers import create_label_distribution_table



def random_forest (dataset, n_trees, sample_size, min_leaf):
     for numbers in range(n_trees):
          create_tree()

def create_tree():
     #get sample_size samples, use np.random.choice w/ replacement
     #create an instance of the Decision Tree class 

def predict():
     #average of predictions from n_trees 
     return binary outcome 0 or 1 