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
    true=np.sum(y_test==y_pred)
    accuracy = true/len(y_test)
    return accuracy

def custom_precision_score(y_true, y_pred):
    """
    This function calculates the precision score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: precision score
    """
    class_zero_precision = np.sum((y_test==0) & (y_pred==0))/np.sum(y_pred==0)
    class_one_precision = np.sum((y_test==1) & (y_pred==1))/np.sum(y_pred==1)
    precision = (class_zero_precision + class_one_precision)/2
    return precision

def custom_recall_score(y_true, y_pred):
    """
    This function calculates the recall score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: recall score
    """
    class_zero_recall = np.sum((y_test==0) & (y_pred==0))/np.sum(y_test==0)
    class_one_recall = np.sum((y_test==1) & (y_pred==1))/np.sum(y_test==1)
    recall_score = (class_zero_recall+class_one_recall)/2
    return recall_score

def custom_f1_score(y_true, y_pred):
    """
    This function calculates the f1 score of the model
    :param y_true: true values
    :param y_pred: predicted values
    :return: f1 score
    """
    class_zero_f1 = 2/(1/(np.sum((y_test==0) & (y_pred==0))/np.sum(y_test==0)) + 1/(np.sum((y_test==0) & (y_pred==0))/np.sum(y_pred==0)))
    class_one_f1 = 2/(1/(np.sum((y_test==1) & (y_pred==1))/np.sum(y_test==1)) + 1/(np.sum((y_test==1) & (y_pred==1))/np.sum(y_pred==1)))
    f1_score = (class_zero_f1 + class_one_f1)/2
    return f1_score

def plot_confusion_matrix(y_true, y_pred):
    """
    This function plots the confusion matrix
    :param y_true: true values
    :param y_pred: predicted values
    :return: None
    """
    l=[]
    tp = np.sum((y_pred == 1) & (y_test == 1))
    tn = np.sum((y_pred == 0) & (y_test == 0))
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    l.append(tn)
    l.append(fn)
    l.append(fp)
    l.append(tp)
    l=np.array(l)
    l=l.reshape((2,2))
    plt.imshow(l, cmap='Blues')
    for i in range(2):
        for j in range(2):
            plt.annotate(l[j][i], xy=(i,j))
    plt.colorbar()
    plt.xlabel('Predicted Values')
    plt.ylabel('Actual Values')
    plt.title('Confusion Matrix')
    plt.xticks([0,1])
    plt.yticks([0,1])
