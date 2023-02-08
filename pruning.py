from classification_helpers import Node
from classification import DecisionTreeClassifier
from evalutation_functions import accuracy

######################################
# PRUNING
######################################

#decision_tree is a DecisionTreeClassifier object
#it's attribute decision_tree (RENAME) is the root Node of our Node class.
#true labels are the true labels found in the validastion set
def prune(decision_tree, node, accuracy, true_labels, validation_set):

    # Looking to see if any of the children are Nodes
    for child_index in range(len(node.children)): # Looping through the two children of the nNode
        if(isinstance(node.children[child_index], Node)): # If children is a Node
            prune_return_value = prune(node.children[child_index], accuracy ) # Recursive call
            
            if(prune_return_value != None): # If the recursion call returned a label (and not None)
                old_child_value = node.children[child_index] # Save in case you'll want to revert the changes
                node.children[child_index] = prune_return_value # Update your tree, by updatingg the value of the child to its label (chaging Node -> Label)

                if(not_better_accuracy(decision_tree, validation_set, true_labels, accuracy)): 
                    #If new accuracy is lower than previous accuracy, revert the changes
                    node.children[child_index] = old_child_value 
   
    # If both children of the Node are labels:
    if(not isinstance(node.children[0], Node) and not isinstance(node.children[1], Node)):
        # count the value of each label
        max_label = 
        return max_label

    # Return if you don't have two labels
    return None


def not_better_accuracy(decision_tree, validation_set, true_labels, previous_accuracy):
    # Calulate new accuracy
    predicted_labels = decision_tree.predict(validation_set) 
    new_accuracy = accuracy(true_labels, predicted_labels)
    return previous_accuracy > new_accuracy