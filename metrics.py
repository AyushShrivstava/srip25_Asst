import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= Instructions =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
# In this assignment, you will be implementing custom metrics functions to evaluate the classification model.
# The task is to use the created custom metrics functions and compare them with the sklearn's metrics functions.
# You use the ml_model.py or ml_model.ipynb file to compare the metrics. 
# When in doubt please reach out to the mentor and thet will help you.
#
# Reading material: 
# 1. https://www.analyticsvidhya.com/blog/2021/07/metrics-to-evaluate-your-classification-model-to-take-the-right-decisions/
# 2. https://youtu.be/Kdsp6soqA7o?si=xZYnBUb6Tk74tQEX
#
# Note: 
# 1. Please do not change the function signature of the functions
# 2. Use of AI tools is not allowed. Try to write the code on your own.
#
#=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= Instructions =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#


def custom_accuracy_score(y_true, y_pred):
    """
    This function calculates the accuracy score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: accuracy score
    """
    assert y_true.size == y_pred.size, "Size of the True Value Array and Predicted Value Array should be Same"
    
    true_values = np.sum(y_true == y_pred)
    false_values = np.sum(y_true != y_pred)
    
    accuracy = true_values / (true_values + false_values)
    return accuracy

def custom_precision_score(y_true, y_pred, average=None):
    """
    Calculates the precision score with options for None, macro, and micro averaging.
    
    :param y_true: numpy array of true values
    :param y_pred: numpy array of predicted values
    :param average: 'macro', 'micro', or None
    :return: precision score
    """
    assert y_true.size == y_pred.size, "Size of the True Value Array and Predicted Value Array should be the same."

    if average is None:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        false_positives = np.sum((y_pred == 1) & (y_true == 0))

        if (true_positives + false_positives) == 0:
            return 0  # Returns 0 to avoid ZeroDivisionError
        return true_positives / (true_positives + false_positives)

    elif average == 'macro':
        classes = [0,1]
        precisions = []

        for cls in classes:
            true_positives = np.sum((y_pred == cls) & (y_true == cls))
            false_positives = np.sum((y_pred == cls) & (y_true != cls))

            if (true_positives + false_positives) == 0:
                precisions.append(0) # Returns 0 to avoid ZeroDivisionError
            else:
                precisions.append(true_positives / (true_positives + false_positives))
        return np.mean(precisions)

    elif average == 'micro':
        true_positives = np.sum((y_pred == y_true) & (y_pred == y_pred))
        false_positives = np.sum((y_pred != y_true) & (y_pred == y_pred))

        if (true_positives + false_positives) == 0:
            return 0  # Returns 0 to avoid ZeroDivisionError
        return true_positives / (true_positives + false_positives)

    else:
        raise ValueError("Invalid Value for average parameter")

def custom_recall_score(y_true, y_pred, average=None):
    """
    Calculates the recall score with options for None, macro, and micro averaging.
    
    :param y_true: numpy array of true values
    :param y_pred: numpy array of predicted values
    :param average: 'macro', 'micro', or None
    :return: recall score
    """
    assert y_true.size == y_pred.size, "Size of the True Value Array and Predicted Value Array should be the same."

    if average is None:
        true_positives = np.sum((y_pred == 1) & (y_true == 1))
        false_negatives = np.sum((y_pred != 1) & (y_true == 1))

        if (true_positives + false_negatives) == 0:
            return 0  # Returns 0 to avoid ZeroDivisionError
        return true_positives / (true_positives + false_negatives)

    elif average == 'macro':
        classes = [0,1]
        recalls = []

        for cls in classes:
            true_positives = np.sum((y_pred == cls) & (y_true == cls))
            false_negatives = np.sum((y_pred != cls) & (y_true == cls))

            if (true_positives + false_negatives) == 0:
                recalls.append(0)  # Returns 0 to avoid ZeroDivisionError
            else:
                recalls.append(true_positives / (true_positives + false_negatives))
        return np.mean(recalls)

    elif average == 'micro':
        true_positives = np.sum((y_pred == y_true) & (y_true == y_true))
        false_negatives = np.sum((y_pred != y_true) & (y_true == y_true))

        if (true_positives + false_negatives) == 0:
            return 0  # Returns 0 to avoid ZeroDivisionError
        return true_positives / (true_positives + false_negatives)

    else:
        raise ValueError("Invalid value for average parameter")

def custom_f1_score(y_true, y_pred, average=None):
    """
    Calculates the F1 score of the model.
    
    :param y_true: numpy array of true values
    :param y_pred: numpy array of predicted values
    :param average: 'macro', 'micro', or None
    :return: F1 score
    """
    assert y_true.size == y_pred.size, "Size of the True Value Array and Predicted Value Array should be the same."

    if average is None:
        precision = custom_precision_score(y_true, y_pred)
        recall = custom_recall_score(y_true, y_pred)

        if (precision + recall) == 0:
            return 0 # Returns 0 to avoid ZeroDivisionError

        f1_score = (2 * precision * recall) / (precision + recall)
        return f1_score
    
    elif average == 'macro':
        classes = [0,1]
        f1_scores = []

        for cls in classes:
            TP = np.sum((y_true == cls) & (y_pred == cls))
            FP = np.sum((y_true != cls) & (y_pred == cls))
            FN = np.sum((y_true == cls) & (y_pred != cls))

            if (TP + FP) == 0 or (TP + FN) == 0:
                return 0 # Returns 0 to avoid ZeroDivisionError

            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            f1 = 2 * precision * recall / (precision + recall)
            f1_scores.append(f1)
        return np.mean(f1_scores)
    
    elif average == 'micro':
        classes = [0, 1]
        TP, FP, FN = 0, 0, 0

        for cls in classes:
            TP += np.sum((y_true == cls) & (y_pred == cls))
            FP += np.sum((y_true != cls) & (y_pred == cls))
            FN += np.sum((y_true == cls) & (y_pred != cls))

        if (TP + FP) == 0 or (TP + FN) == 0:
            return 0 # Returns 0 to avoid ZeroDivisionError
        
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)

        f1_micro = (2 * precision * recall) / (precision + recall)
        return f1_micro

def plot_confusion_matrix(y_true, y_pred):
    """
    This function plots the confusion matrix
    :param y_true: true values
    :param y_pred: predicted values
    :return: None
    """
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    confusion_matrix = np.array([[TN, FP], [FN, TP]])

    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=True, xticklabels=[0, 1], yticklabels=[0, 1])
    plt.xlabel('Predicted Values')
    plt.ylabel('Ground Truth Values')
    plt.title('Confusion Matrix using Seaborn')
    
    plt.show()