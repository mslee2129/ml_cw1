import numpy as np
from scipy.stats import mode
from create_forest_decision_tree import create_forest_decision_tree
from classification_helpers import predict_value


def random_forest (dataset, testing_set, num_attributes_hyperparameter, num_trees_hyperparameter, max_depth = 10000):
    """ Hyperparameters:
    - number of trees in forests
    - number of random attributes to consider for each tree
        - A good heuristic for classification is to set this hyperparameter to
          the square root of the number of input features
    - depth of the decision trees:
    - sample size = length of the dataset
    """
    np.random.seed(42) 
    prediction_list = []
    sample_size = np.shape(dataset)[0]

    for numbers in range(num_trees_hyperparameter):
        
        #bootstrapping the new_dataset which is made from a subset of the attributes
        indexes = np.random.choice(np.shape(dataset)[0], replace=True, size = sample_size)
        dataset = dataset[indexes,:]

        # def create_forest_decision_tree(dataset, num_attributes_hyperparameter ,max_depth = 10000, depth = -1):
        tree = create_forest_decision_tree(dataset, num_attributes_hyperparameter, max_depth)
    
        predictions = np.zeros((testing_set.shape[0],), dtype=np.object_)
        for index in range(testing_set.shape[0]): #Going through every value we want to predict
            #print(index)
            #print(testing_set[index])
            predictions[index] = predict_value(tree, testing_set[index])
        
        prediction_list.append(predictions)
        
    predictions = np.column_stack(tuple(prediction_list)) # now each column contains a different model's prediction for a row
    rf_predictions, _ = mode(predictions, axis=1)
    return np.squeeze(rf_predictions)
    