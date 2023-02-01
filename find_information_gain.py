from find_entropy import find_entropy
import numpy as np
import classification as cl

def find_information_gain(attribute_values, labels, split_index): # attribute-values = x; labels = y
    entropy_pre_split = find_entropy(labels)
   
    data_concat = np.concatenate((attribute_values, np.expand_dims(labels, axis=0).T), axis=1)
    data_sorted = data_concat[data_concat[:,0].argsort()]
    print(data_sorted[:split_index])
    sorted_labels = data_sorted[:,data_sorted.shape[1] - 1]
  
    labels_left = sorted_labels[:split_index]
    labels_right = sorted_labels[split_index:]

    entropy_left = find_entropy(labels_left)
    entropy_right = find_entropy(labels_right)  
    print (entropy_left)
    print(entropy_right)
    entropy_weighted_average = entropy_left * (len(labels_left)/len(labels)) + entropy_right * (len(labels_right)/len(labels))
    
    information_gain = entropy_pre_split - entropy_weighted_average
    
    return information_gain



