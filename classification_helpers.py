import numpy as np
from math import log2

######################################
# HELPER FOR DATA
######################################

def get_int_labels(str_labels): #Takes in string labels and returns an array with the int equivalents
    classes, int_labels = np.unique(str_labels, return_inverse = True)
    return int_labels

def concat_data_helper(data, labels):
    # Adds the labels as the last column of the dataset
    data_concat = np.concatenate((data, np.expand_dims(labels, axis=0).T), axis=1) 
    # For some reason, the line above changes labels to a np.float64. 
    # This changes the whole array into int64
    data_concat = data_concat.astype(np.int64) 
    return data_concat


#####################################################
# HOW TO MAKE SPLITTING DECISIONS - HELPER FUNCTIONS
#####################################################

def suggest_split_points(attribute_index, data):
    assert attribute_index >= 0 and attribute_index < (len(data[0])-1), \
         "Out of bounds: checking to split point at invalid attribute"

    # Sorting the data based on the attribute
    data_sorted = data[data[:,attribute_index].argsort()]

    suggestions = []
    # starting at 1 because comparing to row - 1; going through all the rows
    for row in range(1, len(data_sorted[:,attribute_index])): 
        
        # True [if the labels are different, or] if the attribute value are different
        if data_sorted[row, attribute_index] != data_sorted[row-1, attribute_index]:  
            #finding the value (avg of the two) at which to do the split
            candidate = (data_sorted[row, attribute_index] + \
                 data_sorted[row - 1, attribute_index]) / 2 
            
            #If suggestions is empty OR if the previous entry is not the same value, then append it
            if len(suggestions) == 0 or suggestions[-1] != candidate: 
                suggestions.append(candidate)
    
    return suggestions

def find_optimal_node(data):
    # optimal_node = (attribute_index, split value, 
    #                   label_distribution_left, label_distribution_right)
    optimal_node = (0, 0, [], []) 
    max_information_gain = -1 

    # loop through attributes # -1 since you want to exclude the label column
    for i in range(len(data[0]) - 1): 

        # loop through suggested split points
        for split_point in suggest_split_points(i, data): 
            information_gain, label_distribution_left, label_distribution_right \
                = find_information_gain(data, split_point, i)
            
            #if it is the new max, update optimal_nove and the max value
            if information_gain > max_information_gain: 
                optimal_node = (i, split_point, label_distribution_left, label_distribution_right) 
                max_information_gain = information_gain

    return optimal_node

def find_entropy(class_labels):
    # a list the size of the number of different class instances
    # used to count occurences of each class
    no_of_class_instances = [0] * len(np.unique(class_labels)) 
    class_instance = list(set(class_labels)) #lists the value of the class instances

    for i in range (len(class_labels)):
        no_of_class_instances[class_instance.index(class_labels[i])] += 1 # add to counter list
    
    entropy = 0
    for i in range (len(no_of_class_instances)):
        ratio = no_of_class_instances[i] / len(class_labels)
        entropy -= ratio * log2(ratio)

    return entropy


def find_information_gain(dataset, split_value, attribute_index):

    labels = dataset[:, -1] # Here we are only selecting the labels column (last one)
  
    entropy_pre_split = find_entropy(labels)
    
    children_datasets = make_split(dataset, attribute_index, split_value)

    labels_left = children_datasets[0][:,-1] # keeping only the labels column of the left dataset
    labels_right = children_datasets[1][:,-1] # keeping only the labels column of the right dataset

    # Finding the distribution of labels in the two datasets
    label_distribution_left = create_label_distribution_table(labels_left)
    label_distribution_right = create_label_distribution_table(labels_right)
    
    entropy_left = find_entropy(labels_left)
    entropy_right = find_entropy(labels_right)  

    num_rows = np.shape(dataset)[0]
    entropy_weighted_average = entropy_left * (len(labels_left) / num_rows) + entropy_right * (len(labels_right) / num_rows)
    
    information_gain = entropy_pre_split - entropy_weighted_average

    return (information_gain, label_distribution_left, label_distribution_right)


######################################
# SPLIT
######################################

def make_split(dataset, attribute_index, split_value):
    data_left = dataset[dataset[:,attribute_index] < split_value]
    data_right = dataset[dataset[:,attribute_index] >= split_value]

    return [data_left, data_right]

######################################
# LABEL DISTRIBUTION HELPER FUNCTIONS
######################################
def create_label_distribution_table(label_array):
    label_distribution = []
    for label in label_array:
        found = False

        for sublist in label_distribution:
            # sublist = [label number, number iterations]
            if sublist[0] == label:
                sublist[1] += 1
                found = True
        
        if not found:
            # appending a sublist with label number and iteration 1
            label_distribution.append([label, 1])
    
    return label_distribution

def find_predominant_label(list_of_lists):
    # loop through label_distribution_full to find the maximum value, and it's associated label
        max_value = -1
        max_label = -1
        for sublist in list_of_lists:
            if(sublist[1] > max_value):
                max_value = sublist[1]
                max_label = sublist[0]

        return max_label


######################################
# NODE CLASS
######################################

class Node:
    def __init__(self, optimal_node):
        attribute_index, split_value, label_distribution_left, label_distribution_right = optimal_node
        self.attribute_index = attribute_index
        self.split_value = split_value
        self.label_distribution_left = label_distribution_left
        self.label_distribution_right = label_distribution_right
        self.children = [None, None]
    
    def add_child(self, node, i):
        self.children[i] = node

    def __str__(self):
        return ("\n Split is happening on attribute: % a, " %self.attribute_index +  " below value: %i \n" % self.split_value)
    
    def recursive_print(self):
        print(self)
        
        for i in range(2):
            if isinstance(self.children[i], Node):
                print(self, "Children: %i" % (i))
                self.children[i].recursive_print()
            else:
                print("Children: %i has label" % (i), self.children[i])

######################################
# RECURSION
######################################

def create_decision_tree(dataset, max_depth = 10000, depth = -1):
    # Update depth; root depth is 0
    depth += 1
    
    labels = dataset[:,- 1] #labels column is the last one
    
    # if only one type of label left or they all have the same attributes , or max_depth
    if len(np.unique(labels)) == 1 or len(np.unique(labels)) == 0 or \
        len(np.unique((dataset[:,:-1]), axis=0)) == 1 or depth == max_depth:
        distribution = create_label_distribution_table(labels)
        return find_predominant_label(distribution)

    optimal_node = find_optimal_node(dataset) 
    node = Node(optimal_node) # creating a new node

    #unpacking because make_split only needs the first three elements
    attribute_index, split_value, unused1, unused2 = optimal_node
    children_datasets = make_split(dataset, attribute_index, split_value)

    # If one of the children datasets is empty, we are returning the predominant label
    if children_datasets[0].shape[0] == 0 or children_datasets[1].shape[0] == 0:
        distribution = create_label_distribution_table(labels)
        return find_predominant_label(distribution)

    for i in range(len(children_datasets)): # 0 or 1
        child_node = create_decision_tree(children_datasets[i], max_depth, depth)
  
        node.add_child(child_node,i) # adding the child nodes to the node created earlier.
    
    return node


######################################
# PREDICTION
######################################

def predict_value(decision_tree, data): 

    # CASE 1 of 2
    if(data[decision_tree.attribute_index] < decision_tree.split_value): # take left
        if(isinstance(decision_tree.children[0], Node)): #if there is another branch
            return predict_value(decision_tree.children[0], data)
        
        else: # if it is a leaf, return the predicted value
            return decision_tree.children[0]

    # CASE 2 of 2
    else: # >=
    #if(data[decision_tree.attribute_index] >= decision_tree.split_value): # take right
        if(isinstance(decision_tree.children[1], Node)): #if there is another branch
            return predict_value(decision_tree.children[1], data)
        
        else: # if it is a leaf, return the predicted value
            return decision_tree.children[1]