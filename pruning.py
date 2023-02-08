import numpy as np
from classification_helpers import Node
from classification_helpers import predict_value
from classification import DecisionTreeClassifier
from evalutation_functions import accuracy
from classification_helpers import find_predominant_label
######################################
# PRUNING
######################################

#decision_tree is a DecisionTreeClassifier object
#it's attribute decision_tree (RENAME) is the root Node of our Node class.
#true labels are the true labels found in the validastion set
def prune(root_node, node, accuracy, true_labels, validation_set):

    # Looking to see if any of the children are Nodes
    for child_index in range(len(node.children)): # Looping through the two children of the nNode
        if(isinstance(node.children[child_index], Node)): # If children is a Node
            prune_return_value = prune(root_node, node.children[child_index], accuracy, true_labels, validation_set ) # Recursive call
            
            if(not isinstance(prune_return_value, Node)): # If the recursion call returned a label (and not a Node)
                old_child_value = node.children[child_index] # Save in case you'll want to revert the changes
                node.children[child_index] = prune_return_value # Update your tree, by updatingg the value of the child to its label (chaging Node -> Label)

                new_accuracy = calculate_accuracy(root_node, validation_set, true_labels)
                if new_accuracy > accuracy: #If higher, update the accuracy
                    accuracy = new_accuracy
                
                else: 
                    #If new accuracy is lower than previous accuracy, revert the changes
                    node.children[child_index] = old_child_value 
                
    # If both children of the Node are labels:
    if(not isinstance(node.children[0], Node) and not isinstance(node.children[1], Node)):
        label_repartition_full = []
        for split_data in (node.label_repartition_left, node.label_repartition_right):
            for i in range(len(split_data)):
                found = False
                for sublist in label_repartition_full:
                    # sublist = [label number, number iterations]
                    if sublist[0] == split_data[i][0]:
                        sublist[1] += split_data[i][1]
                        found = True
                
                if not found:
                    # appending a sublist with label number and iteration 1
                    label_repartition_full.append([split_data[i][0], split_data[i][1]])

        return find_predominant_label(label_repartition_full)

    # Return if you don't have two labels
    return root_node # returns itself

def calculate_accuracy(root_node, validation_set, true_labels):
    predicted_labels = np.zeros((validation_set.shape[0],), dtype=np.object_)
    for index in range(validation_set.shape[0]): #Going through every value we want to predict
        predicted_labels[index] = predict_value(root_node,validation_set[index]) 
    new_accuracy = accuracy(true_labels, predicted_labels)
    return new_accuracy