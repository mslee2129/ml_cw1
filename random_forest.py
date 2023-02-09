import numpy as np
import classification_helpers as ch
from classification_helpers import Node
# from classification_helpers import predict_value
# from classification import DecisionTreeClassifier
from evalutation_functions import accuracy
from classification_helpers import find_predominant_label
from classification_helpers import create_label_distribution_table
from scipy.stats import mode
from classification_helpers import find_optimal_node, make_split

def create_forest_decision_tree(dataset, num_attributes_hyperparameter ,max_depth = 10000, depth = -1):
    # Update depth; root depth is 0
    depth += 1
    
    labels = dataset[:,- 1] #labels column is the last one

    if depth == max_depth:
        #print("EXCEEDED MAX DEPTH")
        distribution = create_label_distribution_table(labels)
        return find_predominant_label(distribution)
    
    # if only one type of label left or they all have the same attributes
    if len(np.unique(labels)) == 1 or len(np.unique(labels)) == 0 or len(np.unique((dataset[:,:-1]), axis=0)) == 1: 
        distribution = create_label_distribution_table(labels)
        return find_predominant_label(distribution)

    # this part is specific to the random forest, we randomly select m attributes to select the split at the node
    num_attributes = np.shape(dataset)[1] - 1 #-1 because we don't want to consider the label column
    selected_attributes = []
    while len(selected_attributes) < num_attributes_hyperparameter:
        proposed_attribute = np.random.randint(0, num_attributes)
        if proposed_attribute not in selected_attributes:
            selected_attributes.append(proposed_attribute)
    
    subset = dataset[:,selected_attributes]
    # returns type (i, split_point, label_distribution_left, label_distribution_right) 
    optimal_node = find_optimal_node(subset)
    # ensure optimal split point and attribute is extracted from function
    optimal_node = (selected_attributes[optimal_node[0]],) + optimal_node[1:]

    
    node = Node(optimal_node) # creating a new node

    #unpacking because make_split only needs the first three elements
    attribute_index, split_value, label_distribution_left, label_distribution_right = optimal_node

    children_datasets = make_split(dataset, (attribute_index, split_value))
    # print('\nWe are splitting dataset: ', dataset)
    # print('\n into: ')
    # print(children_datasets[0])
    # if children_datasets[0].shape[0] == 0:
    #     print("They are equal !!")
    # print('and \n')
    # print(children_datasets[1])
    if children_datasets[0].shape[0] == 0 or children_datasets[1].shape[0] == 0:
        distribution = create_label_distribution_table(labels)
        return find_predominant_label(distribution)
    for i in range(len(children_datasets)): # 0 or 1
        child_node = create_forest_decision_tree(children_datasets[i], num_attributes_hyperparameter, max_depth, depth)
  
        node.add_child(child_node,i) # adding the child nodes to the node created earlier.
    
    return node



def random_forest (dataset, testing_set, num_attributes_hyperparameter, num_trees_hyperparameter, max_depth = 10000):
    """ Hyperparameters:
    - number of trees in forests
    - number of random attributes to consider for each tree
        - A good heuristic for classification is to set this hyperparameter to
          the square root of the number of input features
    - depth of the decision trees:
    - sample size = length of the dataset
    """
    np.random.seed(42) 
    prediction_list = []
    sample_size = np.shape(dataset)[0]

    for numbers in range(num_trees_hyperparameter):
        
        #bootstrapping the new_dataset which is made from a subset of the attributes
        indexes = np.random.choice(np.shape(dataset)[0], replace=True, size = sample_size)
        dataset = dataset[indexes,:]

        # def create_forest_decision_tree(dataset, num_attributes_hyperparameter ,max_depth = 10000, depth = -1):
        tree = create_forest_decision_tree(dataset, num_attributes_hyperparameter, max_depth)
    
        predictions = np.zeros((testing_set.shape[0],), dtype=np.object_)
        for index in range(testing_set.shape[0]): #Going through every value we want to predict
            #print(index)
            #print(testing_set[index])
            predictions[index] = ch.predict_value(tree, testing_set[index])
        
        prediction_list.append(predictions)
        
    predictions = np.column_stack(tuple(prediction_list)) # now each column contains a different model's prediction for a row
    rf_predictions, _ = mode(predictions, axis=1)
    return rf_predictions
    