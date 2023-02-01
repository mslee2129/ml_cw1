import numpy as np
from math import log2
def find_entropy(class_labels):
    no_of_class_instances = [0] * len(np.unique(class_labels))
    for i in range (len(class_labels)):
        no_of_class_instances[int (class_labels[i])]+=1
    entropy = 0
    for i in range (len(no_of_class_instances)):
        ratio = no_of_class_instances[i]/len(class_labels)

        entropy -= ratio*log2(ratio)
    return entropy
