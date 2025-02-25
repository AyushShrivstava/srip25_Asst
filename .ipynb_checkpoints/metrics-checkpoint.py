import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    
    n = y_true.size # Will give us the Length of the Array
    TP, TN, FP, FN = 0, 0, 0, 0 # True Positive, True Negative, False Positive and False Negatives defined

    # Finding TP, TN, FP and FN
    for i in range(n):
        true_value, predicted_value = y_true[i], y_pred[i]
        if true_value == 1:
            if predicted_value == 1:
                TP += 1
            elif predicted_value == 0:
                FN += 1
        elif true_value == 0:
            if predicted_value == 0:
                TN += 1
            elif predicted_value == 1:
                FP += 1
    
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    return accuracy

def custom_precision_score(y_true, y_pred):
    """
    This function calculates the precision score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: precision score
    """
    assert y_true.size == y_pred.size, "Size of the True Value Array and Predicted Value Array should be Same"
    
    true_positives = np.sum((y_pred == 1) & (y_true == 1))
    false_positives = np.sum((y_pred == 1) & (y_true == 0))
    print(true_positives, false_positives)
    if (true_positives + false_positives) == 0:
        return 0 # To avoid ZeroDivisionError we return Zero. Sklearn also does this
    
    precision = true_positives / (true_positives + false_positives)
    return precision

def custom_recall_score(y_true, y_pred):
    """
    This function calculates the recall score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: recall score
    """
    assert y_true.size == y_pred.size, "Size of the True Value Array and Predicted Value Array should be Same"
    
    n = y_true.size # Will give us the Length of the Array
    TP, TN, FP, FN = 0, 0, 0, 0 # True Positive, True Negative, False Positive and False Negatives defined

    # Finding TP, TN, FP and FN
    for i in range(n):
        true_value, predicted_value = y_true[i], y_pred[i]
        if true_value == 1:
            if predicted_value == 1:
                TP += 1
            elif predicted_value == 0:
                FN += 1
        elif true_value == 0:
            if predicted_value == 0:
                TN += 1
            elif predicted_value == 1:
                FP += 1
    
    accuracy = (TP) / (TP + TN + FP + FN)
    return accuracy

def custom_f1_score(y_true, y_pred):
    """
    This function calculates the f1 score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: f1 score
    """
    # Write your code here and remove the pass
    pass

def plot_confusion_matrix(y_true, y_pred):
    """
    This function plots the confusion matrix
    :param y_true: true values
    :param y_pred: predicted values
    :return: None
    """
    # Write your code here and remove the pass
    pass
