import numpy as np
from classification_helpers import Node, find_predominant_label, predict_value
from evalutation_functions import accuracy

######################################
# PRUNING
######################################

def prune(root_node, accuracy, true_labels, validation_set, node=None):
    if node == None:
        node = root_node
    
    # Looking to see if any of the children are Nodes
    for child_index in range(len(node.children)): # Looping through the two children of the nNode
        if(isinstance(node.children[child_index], Node)): # If children is a Node
            
            prune_return_value, maybe_accuracy = prune(root_node, accuracy, true_labels, validation_set, node.children[child_index] ) # Recursive call
            #prune_return_value = prune(root_node, accuracy, true_labels, validation_set, node.children[child_index] ) # Recursive call
            
            if(not isinstance(prune_return_value, Node)): # If the recursion call returned a label (and not a Node)
                
                old_child_value = node.children[child_index] # Save in case you'll want to revert the changes
                node.children[child_index] = prune_return_value # Update your tree, by updatingg the value of the child to its label (chaging Node -> Label)
    
                new_accuracy = calculate_accuracy(root_node, validation_set, true_labels)
                if new_accuracy >= accuracy: #If higher, update the accuracy
                    accuracy = new_accuracy #here the accuracy update only lives from here down. not if we got back up in the tree...
                
                else: 
                    #If new accuracy is lower than previous accuracy, revert the changes
                    node.children[child_index] = old_child_value 
            
            else:
                accuracy = maybe_accuracy
                
    # If both children of the Node are labels:
    if(not isinstance(node.children[0], Node) and not isinstance(node.children[1], Node)):
     
        repartition_left = node.label_distribution_left
        repartition_right = node.label_distribution_right
 
        label_repartition_full = []
        for sublist in repartition_left:
            label_repartition_full.append(sublist)

            for right_sublist in repartition_right:
                if right_sublist[0] == sublist[0]: #if we're not comparing the two same labels
                    label_repartition_full[-1][1] += right_sublist[1] #adding to full the value of the right subset

        
        #return find_predominant_label(label_repartition_full)
        return (find_predominant_label(label_repartition_full), -1)

    # Return if you don't have two labels
    return (root_node, accuracy) # returns itself
    #return root_node # returns itself


def calculate_accuracy(root_node, validation_set, true_labels):
    predicted_labels = np.zeros((validation_set.shape[0],), dtype=np.object_)
    
    for index in range(validation_set.shape[0]): #Going through every value we want to predict
        predicted_labels[index] = predict_value(root_node, validation_set[index]) 
    
    new_accuracy = accuracy(true_labels, predicted_labels)
    
    return new_accuracy