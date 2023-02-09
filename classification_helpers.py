import numpy as np
from math import log2
import math
import matplotlib.pyplot as plt

######################################
# Read and examine
######################################

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
    
    # Showing a graphical representation of the distribution (#) of observations by classes
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

    # Showing a graphical representation of the distribution (%) of observations by classes
    plt.bar(range(num_classes), ratio, )
    title_str = "Ratio of dataset: " + dataset_name
    plt.title(title_str)
    plt.xlabel('Class number')
    plt.ylabel('Percentage of values (weight of the class)')
    # plt.show()
    file_name = "graphs/ratio_" + dataset_name
    plt.savefig(file_name)
    plt.clf() # Clears the figure so the graphs don't overlap in the saved file

def noisy_data_comparison(clean_data, noisy_data):
    # sort both arrays by attributes, not the label
    # we then iterate through the classes to see which observations differ

    # iteratively sort each attribute, see: 
    # https://opensourceoptions.com/blog/sort-numpy-arrays-by-columns-or-rows/#:~:text=NumPy%20arrays%20can%20be%20sorted,an%20array%20in%20ascending%20value.
    # print('Num attributes: ', len(clean_data[0])-1)
    num_attributes = len(clean_data[0])-1
    for i in reversed(range(num_attributes)):
        if i == num_attributes-1:
            clean_data = clean_data[clean_data[:,i].argsort()]
            noisy_data = noisy_data[noisy_data[:,i].argsort()]
        else:
            clean_data = clean_data[clean_data[:,i].argsort(kind='mergesort')]
            noisy_data = noisy_data[noisy_data[:,i].argsort(kind='mergesort')]
    
    num_observation = len(clean_data[:,0])
    matches = 0

    # print('clean sorted data: ', clean_data[:10, :], '\n')
    # print('noisy sorted data: ', noisy_data[:10, :], '\n')

    for i in range(num_observation):
        # print('clean: ', clean_data[i,-1], ' noisy: ', noisy_data[i,-1], '\n')
        if clean_data[i,-1] == noisy_data[i,-1]:
            matches += 1
    print('Matches (%): ', matches / num_observation)
    print('Incorrect observations in noisy data (%): ', 1 - (matches / num_observation), '\n')

























######################################
# Helpers for recursion
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


def suggest_split_points(attribute_index, data):
    assert attribute_index >= 0 and attribute_index < (len(data[0])-1), "Out of bounds: checking to split point at invalid attribute"

    # Sorting the data based on the attribute
    data_sorted = data[data[:,attribute_index].argsort()]

    suggestions = []
    # starting at 1 because comparing to row - 1; going through all the rows
    for row in range(1, len(data_sorted[:,attribute_index])): 
        
        # True [if the labels are different, or] if the attribute value are different
        # if data_sorted[row, -1] != data_sorted[row-1, -1] or data_sorted[row, attribute_index] != data_sorted[row-1, attribute_index]:  
        # if data_sorted[row, attribute_index] != data_sorted[row-1, attribute_index]:  
        if data_sorted[row, -1] != data_sorted[row-1, -1]:
            #finding the value (avg of the two) at which to do the split
            candidate = (data_sorted[row, attribute_index] + data_sorted[row - 1, attribute_index]) / 2 
            
            #If suggestions is empty OR if the previous entry is not the same value, then append it
            if len(suggestions) == 0 or suggestions[-1] != candidate: 
                suggestions.append(candidate)
    
    #print("Suggestions for attribute %a:" % attribute_index, suggestions)
    return suggestions



def find_optimal_node(data):
    # optimal_node = (attribute_index, split value, label_distribution_left, label_distribution_right)
    optimal_node = (0, 0, [], []) 
    max_information_gain = -1 

    # loop through attributes # -1 since you want to exclude the label column
    for i in range(len(data[0]) - 1): 

        # loop through suggested split points
        for split_point in suggest_split_points(i, data): 
            information_gain, label_distribution_left, label_distribution_right = find_information_gain(data, split_point, i)
            
            #if it is the new max, update optimal_nove and the max value
            if information_gain > max_information_gain: 
                optimal_node = (i, split_point, label_distribution_left, label_distribution_right) 
                max_information_gain = information_gain

    return optimal_node



def find_entropy(class_labels):
    # a list the size of the number of different class instances
    no_of_class_instances = [0] * len(np.unique(class_labels))  # used to count occurences of each class
    class_instance = list(set(class_labels)) #lists the value of the class instances

    for i in range (len(class_labels)):
        no_of_class_instances[class_instance.index(class_labels[i])] += 1 # add to counter list
    
    entropy = 0
    for i in range (len(no_of_class_instances)):
        ratio = no_of_class_instances[i] / len(class_labels)
        entropy -= ratio * log2(ratio)

    return entropy


def find_information_gain(dataset, split_value, attribute_index): # attribute-values = x; labels = y

    labels = dataset[:, -1] # Here we are only selecting the labels column (last one)
  
    entropy_pre_split = find_entropy(labels)
    
    
    split_info = (attribute_index,split_value) # POTENTIALLY remove split_info towards THE END
    children_datasets = make_split(dataset, split_info)

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



def make_split(dataset, split_info):

    attribute_index, split_value = split_info
    data_left = dataset[dataset[:,attribute_index] < split_value]
    data_right = dataset[dataset[:,attribute_index] >= split_value]
    return [data_left, data_right]

######################################
# RECURSION
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
     

def find_predominant_label(list_of_lists):
    # loop through label_distribution_full to find the maximum value, and it's associated label
        max_value = -1
        max_label = -1
        for sublist in list_of_lists:
            if(sublist[1] > max_value):
                max_value = sublist[1]
                max_label = sublist[0]

        return max_label

def create_decision_tree(dataset, max_depth = 10000, depth = -1):
    # Update depth; root depth is 0
    depth += 1
    
    labels = dataset[:,- 1] #labels column is the last one

    if depth == max_depth:
        print("EXCEEDED MAX DEPTH")
        distribution = create_label_distribution_table(labels)
        return find_predominant_label(distribution)
    
    # if only one type of label left or they all have the same attributes
    if len(np.unique(labels)) == 1 or len(np.unique(labels)) == 0 or len(np.unique((dataset[:,:-1]), axis=0)) == 1: 
        distribution = create_label_distribution_table(labels)
        return find_predominant_label(distribution)

    optimal_node = find_optimal_node(dataset) 
    
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