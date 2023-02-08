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
def prune(root_node, accuracy, true_labels, validation_set, node=None):
    if node == None:
        node = root_node
    
    if(node is root_node):
        print("OMG WE ARE THE SAME\n")
    #print("root:",id(root_node))
    # Looking to see if any of the children are Nodes
    for child_index in range(len(node.children)): # Looping through the two children of the nNode
        if(isinstance(node.children[child_index], Node)): # If children is a Node
            
            #prune_return_value, maybe_accuracy = prune(root_node, accuracy, true_labels, validation_set, node.children[child_index] ) # Recursive call
            prune_return_value = prune(root_node, accuracy, true_labels, validation_set, node.children[child_index] ) # Recursive call
            
            
            if(not isinstance(prune_return_value, Node)): # If the recursion call returned a label (and not a Node)
                
                #print(node.children[child_index])
                old_child_value = node.children[child_index] # Save in case you'll want to revert the changes
                node.children[child_index] = prune_return_value # Update your tree, by updatingg the value of the child to its label (chaging Node -> Label)
                #print(node.children[child_index])
                                                ## The issue must be that root_node here is not being updated........
                new_accuracy = calculate_accuracy(root_node, validation_set, true_labels)
                if new_accuracy > accuracy: #If higher, update the accuracy
                    #print("UPDATED ----------------------------------------------------------------------------------------")
                    print(accuracy, "-->", new_accuracy)

                    #print("OLD CHILD:", old_child_value)
                    #print("OLD CHILD'S CHILDREN:", old_child_value.children)
                    #print("NEW CHILD:", prune_return_value)

                    accuracy = new_accuracy #here the accuracy update only lives from here down. not if we got back up in the tree...
                
                else: 
                    print("THIS WAS FALSE")
                    #If new accuracy is lower than previous accuracy, revert the changes
                    node.children[child_index] = old_child_value 
            
            #else:
            #    accuracy = maybe_accuracy
                
    # If both children of the Node are labels:
    if(not isinstance(node.children[0], Node) and not isinstance(node.children[1], Node)):
        #print("!YOHO!")
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

        return (find_predominant_label(label_repartition_full), -1)

    # Return if you don't have two labels
    #print("In the end, I am", id(node))
    #return (root_node, accuracy) # returns itself
    return root_node # returns itself


def calculate_accuracy(root_node, validation_set, true_labels):
    predicted_labels = np.zeros((validation_set.shape[0],), dtype=np.object_)
    for index in range(validation_set.shape[0]): #Going through every value we want to predict
        predicted_labels[index] = predict_value(root_node, validation_set[index]) 
    new_accuracy = accuracy(true_labels, predicted_labels)
    #print(":) ", new_accuracy)
    return new_accuracy