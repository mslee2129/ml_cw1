import numpy as np
from classification import read_dataset
from find_information_gain import find_information_gain
"""
INPUTS:
- index, i, representing the attribute index
- dataset

OUTPUTS:
- 
"""

data = read_dataset("./data/toy.txt")

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
            candidate = np.floor((data_sorted[i,idx] + data_sorted[i-1,idx])/2)
            if len(suggestions) == 0 or suggestions[-1] != candidate:
                suggestions.append(candidate)
    print(suggestions)
    return suggestions

def find_optimal_node(data):
    optimal_node = (0, 0, 0) # optimal_node = (index, split value, information gain)
    for i in range(len(data[0])-1): # loop through attributes
        for split_point in suggest_split_points(i, data): # loop through suggested split points
            information_gain = find_information_gain(data, i)
            if information_gain > optimal_node[2]:
                optimal_node = (i, split_point, information_gain)
    print("The optimal node is: ", optimal_node)
    return optimal_node

find_optimal_node(concat_data_helper(data))