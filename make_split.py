import numpy as np
def make_split(attribute_values, labels, split_index): 
    data_concat = np.concatenate((attribute_values, np.expand_dims(labels, axis=0).T), axis=1) #sorting the dataset
    data_sorted = data_concat[data_concat[:,0].argsort()]




   
