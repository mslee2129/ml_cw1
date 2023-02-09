import numpy as np
from classification_helpers import get_int_labels
######################################
# EVALUATION
######################################

def print_all_evaluation_metrics(str_y_gold, y_prediction):
    
    # Getting int labels, instead of str
    y_gold = get_int_labels(str_y_gold)
    if(isinstance(y_prediction[0], str)):
        y_prediction = get_int_labels(y_prediction)
    

    matrix = confusion_matrix(y_gold, y_prediction)
    acc_results = accuracy(y_gold, y_prediction)
    prec_results, macro_prec = precision(y_gold, y_prediction, matrix)
    recc_results, macro_recc = recall(y_gold, y_prediction, matrix)
    f1_results, macro_f1 = f1_score(y_gold, y_prediction, matrix)

    print("\n Confusion matrix:\n", matrix, "\n")

    print("Precision results per label type:", prec_results)
    print("Recall results per label type:", recc_results)
    print("F1 results per label type:", f1_results)
    
    print("\nAccuracy results:", acc_results)  
    print("Macro Precision:", macro_prec)
    print("Macro Recall:", macro_recc)
    print("Macro F1:", macro_f1)

def confusion_matrix(y_gold, y_prediction, class_labels=None):
    """ Compute the confusion matrix.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels
        class_labels (np.ndarray): a list of unique class labels. 
                               Defaults to the union of y_gold and y_prediction.

    Returns:
        np.array : shape (C, C), where C is the number of classes. 
                   Rows are ground truth per class, columns are predictions
    """

    # if no class_labels are given, we obtain the set of unique class labels from
    # the union of the ground truth annotation and the prediction
    if not class_labels:
        class_labels = np.unique(np.concatenate((y_gold, y_prediction)))

    confusion = np.zeros((len(class_labels), len(class_labels)), dtype=np.int_)

    for r in range(len(y_prediction)):
      confusion[y_gold[r]][y_prediction[r]] += 1

    return confusion


def accuracy(y_gold, y_prediction):
    """ Compute the accuracy given the ground truth and predictions

    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        float : the accuracy
    """

    assert len(y_gold) == len(y_prediction)  
    
    try:
        return np.sum(y_gold == y_prediction) / len(y_gold)
    except ZeroDivisionError:
        return 0.

def precision(y_gold, y_prediction, matrix):
    """ Compute the precision score per class given the ground truth and predictions
        
    Also return the macro-averaged precision across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (precisions, macro_precision) where
            - precisions is a np.ndarray of shape (C,), where each element is the 
              precision for class c
            - macro-precision is macro-averaged precision (a float) 
    """
    # Compute the precision per class
    rows, cols = np.shape(matrix)
    
    p = []
    for x in range(len(np.unique(y_gold))):
      num = 0
      denom = 0

      for row in range(rows):
        for col in range(cols):
          if row == x and col == x:
            num = matrix[row][col]
            denom += matrix[row][col]

          elif col == x:
            denom += matrix[row][col]
            
      p.append(num / denom)
    

    # Compute the macro-averaged precision
    macro_p = sum(p) / len(p)
    
    p = tuple(p)
    return (p, macro_p)


def recall(y_gold, y_prediction, matrix):
    """ Compute the recall score per class given the ground truth and predictions
        
    Also return the macro-averaged recall across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (recalls, macro_recall) where
            - recalls is a np.ndarray of shape (C,), where each element is the 
                recall for class c
            - macro-recall is macro-averaged recall (a float) 
    """
    # Compute the precision per class
    rows, cols = np.shape(matrix)
    
    r = []
    for x in range(len(np.unique(y_gold))):
      num = 0
      denom = 0

      for row in range(rows):
        for col in range(cols):
          if row == x and col == x:
            num = matrix[row][col]
            denom += matrix[row][col]

          elif row == x:
            denom += matrix[row][col]
            
      r.append(num / denom)
    

    # Compute the macro-averaged precision
    macro_r = sum(r) / len(r)
    
    r = tuple(r)
    return (r, macro_r)

def f1_score(y_gold, y_prediction, matrix):
    """ Compute the F1-score per class given the ground truth and predictions
        
    Also return the macro-averaged F1-score across classes.
        
    Args:
        y_gold (np.ndarray): the correct ground truth/gold standard labels
        y_prediction (np.ndarray): the predicted labels

    Returns:
        tuple: returns a tuple (f1s, macro_f1) where
            - f1s is a np.ndarray of shape (C,), where each element is the 
              f1-score for class c
            - macro-f1 is macro-averaged f1-score (a float) 
    """
    (precisions, macro_p) = precision(y_gold, y_prediction, matrix)
    (recalls, macro_r) = recall(y_gold, y_prediction, matrix)

    # just to make sure they are of the same length
    assert len(precisions) == len(recalls)

    # Complete this to compute the per-class F1
    f = np.zeros((len(precisions), ))

    for i in range(len(precisions)):
      f[i] = (2 * precisions[i] * recalls[i] / (precisions[i] + recalls[i]))

    # Compute the macro-averaged F1
    macro_f = np.sum(f) / len(f)
    
    return (f, macro_f)