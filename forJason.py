import numpy as np
import classification_helpers as ch
from evalutation_functions import print_all_evaluation_metrics, accuracy
from pruning import prune
from random_forest import random_forest
from load_dataset import load_dataset
from classification import DecisionTreeClassifier


def predict(tree, set):
    predictions = np.zeros((set.shape[0],), dtype=np.object_)    
    for index in range(set.shape[0]): #Going through every value we want to predict
        predictions[index] = ch.predict_value(tree, set[index])
    return predictions


x_full,y_full,c_full  = load_dataset("./data/train_full.txt")
data = ch.concat_data_helper(x_full,y_full)

x_noisy,y_noisy,c_noisy  = load_dataset("./data/train_noisy.txt")
data_noisy = ch.concat_data_helper(x_noisy,y_noisy)

x_val, y_val, class_val  = load_dataset("./data/validation.txt")
x_test, y_test, class_test  = load_dataset("./data/test.txt")


print("##############################################")
print("# ORIGINAL / Q2 TREE")
print("##############################################")

print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON FULL_DATASET ")
full_classifier = DecisionTreeClassifier()
full_classifier.fit(x_full, y_full)
full_predictions = full_classifier.predict(x_test)
print_all_evaluation_metrics(y_test, full_predictions)
full_classifier_accuracy = accuracy(y_test, full_predictions)


print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON NOISY_DATASET ")
noisy_classifier = DecisionTreeClassifier()
noisy_classifier.fit(x_noisy, y_noisy)
noisy_predictions = noisy_classifier.predict(x_test)
print_all_evaluation_metrics(y_test, noisy_predictions)
noisy_classifier_accuracy = accuracy(y_test, noisy_predictions)



print("##############################################")
print("# Pruning on ORIGINAL")
print("##############################################")

print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON FULL_DATASET ")
pruned_tree, _ = prune(full_classifier.decision_tree, full_classifier_accuracy, y_val, x_val)
predictions = predict(pruned_tree, x_test)
print_all_evaluation_metrics(y_test, predictions)

print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON NOISY_DATASET ")
pruned_tree, _ = prune(noisy_classifier.decision_tree, noisy_classifier_accuracy, y_val, x_val)
predictions = predict(pruned_tree, x_test)
print_all_evaluation_metrics(y_test, predictions)




print("##############################################")
print("# Max Depth = 13")
print("##############################################")

print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON FULL_DATASET ")
max_decision_tree = ch.create_decision_tree(data, 13)
max_predictions = predict(max_decision_tree, x_test)
print_all_evaluation_metrics(y_test, max_predictions)
max_tree_accuracy = accuracy(y_test, max_predictions)

print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON NOISY_DATASET ")
noisy_max_decision_tree = ch.create_decision_tree(data_noisy, 13)
noisy_max_predictions = predict(noisy_max_decision_tree, x_test)
print_all_evaluation_metrics(y_test, noisy_max_predictions)
noisy_max_tree_accuracy = accuracy(y_test, noisy_max_predictions)



print("##############################################")
print("# Pruning on Max Depth")
print("##############################################")

print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON FULL_DATASET ")
pruned_tree, _ = prune(max_decision_tree, max_tree_accuracy, y_val, x_val)
predictions = predict(pruned_tree, x_test)
print_all_evaluation_metrics(y_test, predictions)

print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON NOISY_DATASET ")
pruned_tree, _ = prune(noisy_max_decision_tree, noisy_max_tree_accuracy, y_val, x_val)
predictions = predict(pruned_tree, x_test)
print_all_evaluation_metrics(y_test, predictions)


print("##############################################")
print("# Random Forest: 71, 5")
print("##############################################")

print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON FULL_DATASET ")
rf_predictions = random_forest(data, x_test, 5, 71)
print_all_evaluation_metrics(y_test, rf_predictions)

print("\n PERFORMANCE ON TEST_DATA WITH TRAINING ON NOISY_DATASET ")
noisy_rf_predictions = random_forest (data_noisy, x_test, 5, 71)
print_all_evaluation_metrics(y_test, noisy_rf_predictions)