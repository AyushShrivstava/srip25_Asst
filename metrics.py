import numpy as np
import pandas as pd
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
    true_vals_count = np.sum(y_pred == y_true)
    return true_vals_count/len(y_true)       # len(y_true) will return the total number of elements in y_true, which is equal to the sum of the counts of all true values and false values
    

def custom_precision_score(y_true, y_pred):
    """
    This function calculates the precision score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: precision score
    """
    classes = [0, 1]
    precision_sum = 0
    
    for current_class in classes:
        TP = np.sum((y_pred == current_class) & (y_true == current_class))
        FP = np.sum((y_pred == current_class) & (y_true != current_class))
        
        if (TP + FP) != 0:
            precision_sum += TP/(TP + FP)
        
    return precision_sum/len(classes)
        

def custom_recall_score(y_true, y_pred):
    """
    This function calculates the recall score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: recall score
    """
    classes = [0, 1]
    recall_sum = 0
    
    for current_class in classes:
        TP = np.sum((y_pred == current_class) & (y_true == current_class))
        FN = np.sum((y_pred != current_class) & (y_true == current_class))
        if (TP + FN) != 0:
            recall_sum += TP/(TP + FN)
        
    return recall_sum/len(classes)

def custom_f1_score(y_true, y_pred):
    """
    This function calculates the f1 score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: f1 score
    """
    classes = [0, 1]
    f1_sum = 0
    
    for current_class in classes:
        TP = np.sum((y_pred == current_class) & (y_true == current_class))
        FP = np.sum((y_pred == current_class) & (y_true != current_class))
        FN = np.sum((y_pred != current_class) & (y_true == current_class))
        
        precision = 0
        recall = 0
        
        if (TP + FP) > 0:
            precision = TP / (TP + FP)
        if (TP + FN) > 0:
            recall = TP / (TP + FN)
        
        if (precision + recall) > 0:
            f1_sum += 2 * precision * recall / (precision + recall)
        
    return f1_sum / len(classes)

def plot_confusion_matrix(y_true, y_pred):
    """
    This function plots the confusion matrix
    :param y_true: true values
    :param y_pred: predicted values
    :return: None
    """
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))
    
    matrix = np.array([[TN, FP], [FN, TP]])
    
    sns.heatmap(matrix, cmap='Blues', annot=True, fmt='d')
    plt.title('Custom Confusion Matrix')
    plt.xlabel("Model Predicted Values")
    plt.ylabel("Ground Truth Values")
    plt.xticks(label=[0, 1])
    plt.yticks(label=[0, 1])
    plt.show()