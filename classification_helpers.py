import numpy as np
from math import log2
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



######################################
# Helpers for recursion
######################################

def concat_data_helper(x, y):
    # print('Our x values are: ', x)
    # print('Our y values are: ', y)
    # print('Our classes are: ', classes)
    # print('Y transpose: ', np.expand_dims(y, axis=0).T)
    data_concat = np.concatenate((x, np.expand_dims(y, axis=0).T), axis=1)
    return data_concat


def suggest_split_points(idx, data):
    assert idx >= 0 and idx < (len(data[0])-1), "Out of bounds: checking to split point at invalid attribute"
    data_sorted = data[data[:,idx].argsort()]
    # print("Sorted data: ", data_sorted)
    suggestions = []
    for i in range(1,len(data_sorted[:,0])):
        if data_sorted[i,-1] != data_sorted[i-1,-1] :
            candidate = int ((data_sorted[i,idx] + data_sorted[i-1,idx]) / 2)
            if len(suggestions) == 0 or suggestions[-1] != candidate:
                suggestions.append(candidate)
    
    #print("Suggestions for attribute %a:" % idx, suggestions)
    return suggestions



def find_optimal_node(data):
    optimal_node = (0, 0) # optimal_node = (index, split value)
    max_information_gain = 0

    for i in range(len(data[0]) - 1): # loop through attributes # -1 since you want to exclude the label column

        for split_point in suggest_split_points(i, data): # loop through suggested split points
            information_gain = find_information_gain(data, split_point, i)

            #print("Hey Hey, I am trying to find the information gain of split point", split_point, "on attribute", i, "and i found", information_gain)
            
            if information_gain > max_information_gain:
                optimal_node = (i, split_point)
             
                max_information_gain = information_gain
            

    #print("The optimal node is: ", optimal_node, "with information gain:", max_information_gain)

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
    
    #print("$$$", entropy)
    return entropy



def find_information_gain(dataset, split_index, attribute_index): # attribute-values = x; labels = y

    # data_concat = np.concatenate((attribute_values, np.expand_dims(labels, axis=0).T), axis=1)
    # data_sorted = dataset[dataset[:,attribute_index].argsort()] # add attribute_index
    
    rows, cols = np.shape(dataset)
    labels = dataset[:,cols - 1] # -1 because shape returns the length
  
    entropy_pre_split = find_entropy(labels)
    #print("!:) :( ", entropy_pre_split)

    left = dataset[dataset[:,attribute_index] <= split_index]
    labels_left = left[:,-1]
    right = dataset[dataset[:,attribute_index] > split_index]
    labels_right = right[:,-1]

    #print("Split index: ", split_index, "on attribute", attribute_index)
    #print("DATABSE: ", dataset)
    #print("LEFT!!!!!!!!!!!!!!!!!!!!!" , left)    
    #print("LEFT LABELS!!!!!" , labels_left)   
    #print("RIGHT!!!!!!!!!!!!!!!!!!!!!" , right)    
    #print("RIGGHT LABELS!!!!!!" , labels_right)    

    #labels_left = labels[:split_index]
    #labels_right = labels[split_index:]

    entropy_left = find_entropy(labels_left)
    entropy_right = find_entropy(labels_right)  

    entropy_weighted_average = entropy_left * (len(labels_left) / rows) + entropy_right * (len(labels_right) / rows)
    
    information_gain = entropy_pre_split - entropy_weighted_average
    print(information_gain)
    return information_gain


def make_split(dataset, attribute_index, split_index): 
    # data_sorted = dataset[dataset[:,attribute_index].argsort()]
    data_left = dataset[dataset[:,attribute_index] <= split_index]
    data_right = dataset[dataset[:,attribute_index] > split_index]
    
    return [data_left, data_right]


######################################
# RECURSION
######################################

class Node:
    def __init__(self, attribute_index, split_index):
        self.attribute_index = attribute_index
        self.split_index = split_index
        self.children = [None, None]
    
    def add_child(self, node, i):
        self.children[i] = node

    def __str__(self):
        return "Split is happening at attribute: % a and index: %i" % (self.attribute_index, self.split_index)
    
    def recursive_print(self):
        print(self)
        
        for i in range(2):
            if isinstance(self.children[i], Node):
                print(self, "Children: %i" % (i))
                self.children[i].recursive_print()
            else:
                print("Children: %i has label" % (i), self.children[i])
     

def create_decision_tree(dataset):
    rows, cols = np.shape(dataset)

    labels = dataset[:,cols - 1] #labels column is the last one
    #print("Dataset :", dataset)
    #print("Labels :", labels)
    #print(len(np.unique((dataset[:,:-1]), axis=0)))

    if len(np.unique(labels)) == 1 or len(np.unique((dataset[:,:-1]), axis=0)) == 1: # if only one type of label left or they all have the same attributes
        return labels[0]

    #print("I wasn't caught")

    attribute_index, split_index = find_optimal_node(dataset) # maybe want to merge with line below, but would need to modify the input and return parameters of find_optimal_node and make_split
   
    #print("Split on attribute", attribute_index, "at index", split_index)
   
    node = Node(attribute_index, split_index)
    children_datasets = make_split(dataset, attribute_index, split_index)
   
    for i in range(len(children_datasets)): # 0 or 1
        print(children_datasets[i])
        child_node = create_decision_tree(children_datasets[i])
  
        node.add_child(child_node,i) 
    
    return node