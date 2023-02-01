import numpy as np
from classification import read_dataset
from math import log2

def concat_data_helper(data):
    x, y, classes = data
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
        if data_sorted[i,-1] != data_sorted[i-1,-1]:
            candidate = int ((data_sorted[i,idx] + data_sorted[i-1,idx]) / 2)
            if len(suggestions) == 0 or suggestions[-1] != candidate:
                suggestions.append(candidate)
    # print(suggestions)
    return suggestions



def find_optimal_node(data):
    optimal_node = (0, 0) # optimal_node = (index, split value)
    max_information_gain = 0
    for i in range(len(data[0])-1): # loop through attributes
        for split_point in suggest_split_points(i, data): # loop through suggested split points
            information_gain = find_information_gain(data, int(split_point), i)
            if information_gain > max_information_gain:
                optimal_node = (i, split_point)
                max_information_gain = information_gain

    print("The optimal node is: ", optimal_node, "with information gain:", max_information_gain)
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



def find_information_gain(dataset, split_index, attribute_index): # attribute-values = x; labels = y

    # data_concat = np.concatenate((attribute_values, np.expand_dims(labels, axis=0).T), axis=1)
    data_sorted = dataset[dataset[:,attribute_index].argsort()] # add attribute_index
    
    rows, cols = np.shape(dataset)
    sorted_labels = data_sorted[:,cols - 1] # -1 because shape returns the length
  
    entropy_pre_split = find_entropy(sorted_labels)

    labels_left = sorted_labels[:split_index]
    labels_right = sorted_labels[split_index:]

    entropy_left = find_entropy(labels_left)
    entropy_right = find_entropy(labels_right)  

    entropy_weighted_average = entropy_left * (len(labels_left) / rows) + entropy_right * (len(labels_right) / rows)
    
    information_gain = entropy_pre_split - entropy_weighted_average
    
    return information_gain


def make_split(dataset, attribute_index, split_index): 
    data_sorted = dataset[dataset[:,attribute_index].argsort()]
    data_left = data_sorted[:split_index]
    data_right = data_sorted[split_index:]
    
    return [data_left, data_right]
