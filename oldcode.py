import time
import numpy as np
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
import csv
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report

def knnErrors( X_data, y_data, Optimal_K ):
    '''
    Determine the accuracy of predicting X_data with a K Nearest Neighbor 
    Classifier trained with most optimal number of neighbors
    

    Parameters
    ----------
    X_data : Data to be predicted
    y_data : Target class values
    Optimal_K : Most Optimal value of K

    Returns
    -------
    scoreAccuracy: Number representing the accuracy of the trainer on X_data and y_data

    '''
    # Build KNN Classifier with an optimal value of K 
    knn_clf = build_NearrestNeighbours_classifier( X_training, y_training, numberOfNeighbors = Optimal_K )
    
    # Predict X_data using build classifier and store it in variable y_pred
    y_pred = knn_clf.predict( X_data ) 
    
    # Calculate Score of Accuracy by taking difference between targeted value and predicted value
    scoreAccuracy = metrics.accuracy_score( y_data, y_pred )
    
    # Return Value of Score Accuracy
    return scoreAccuracy
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
def svmErrors(X_data, y_data, C):
    '''
    
    Determine the accuracy of predicting X_data with a Support Vector Machine 
    Classifier with the most optimal C value

    Parameters
    ----------
    X_data : Data to be predicted
    y_data : Target class values
    C: Most Optimal value of C

    Returns
    -------
    scoreAccuracy: Number representing the accuracy of the trainer on X_data and y_data


    '''
    # Build SVM Classifier with an optimal value of C 
    svm_clf = build_SupportVectorMachine_classifier( X_training, y_training, boxConstraint = C )
    
    # Predict X_data using build classifier and store it in variable y_pred
    y_pred = svm_clf.predict( X_data )
    
    # Calculate Score of Accuracy by taking difference between targeted value and predicted value
    scoreAccuracy = metrics.accuracy_score( y_data, y_pred )
    
    # Return Value of Score Accuracy
    return scoreAccuracy
    
        
def knnTests(X_data, y_data, neighbours):
    
    '''
    
    Determine the most optimal number of neighbours for K Nearest Neighbour Classifier
    which classifies the training data with least Mean Sqaured Error (MSE)

    @param 
	X_data: 30 Features data for each record 
	y_data: Target label for each record

    @return
	optimal_K : number representing the most optimal value for number of neighbours
    
    '''
    # Initialising an empty array to store Mean Squared Errors
    MSE = []
    
    # Loop over given array to calcukate Cross Val Score and MSE for each trained model with different number of neighbors
    for k in neighbours:
        
        # Build KNN classifier
        knn = build_NearrestNeighbours_classifier( X_training, y_training ) 
        
        # Calculate Cross Val Score
        scores = cross_val_score( knn, X_data, y_data, cv = 5, scoring = 'accuracy' )
        
        # Calculate MSE from Cross Val Score
        MSE.append( 1 - scores.mean() )
        
    # Find where MSE is minimum    
    minIndexMSE = MSE.index(min(MSE)) 
    
    # Find corresponding Neighbour value
    optimal_K = neighbours[minIndexMSE]
    
    print ("The optimal number of neighbours is %d" % optimal_K)
    neighbours = list(neighbours)
    
    
    #Plot MSE vs Neighbor
    plt.plot(neighbours, MSE)
    plt.xlabel('Number of Neighbours K')
    plt.ylabel('Misclassification Error')
    plt.show()
    
    # Return Optimal number of Neighbours
    return optimal_K


def svmTests(X_data, y_data, cValues):
    
    '''
    Determine the most optimal value of C for Support Vector Machine Classifier
    which classifies the training data with least Mean Sqaured Error (MSE)

    @param 
	X_data: 30 Features data for each record 
	y_data: Target label for each record

    @return
	optimal_C : number representing the most optimal C Value
    
    '''
    
    #Initialising empty array to store Mean Squared Errors
    MSE = []
    
    # Loop over given array to calcukate Cross Val Score and MSE for each trained model with different C value
    for i in cValues:
        
        # Build SVM classifier
        clf = build_SupportVectorMachine_classifier(X_training, y_training, parameter_C = i )
        
        # Calculate Cross Val Score
        scores = cross_val_score(clf, X_data, y_data, cv=5)
        
        # Calculate MSE from Cross Val Score
        MSE.append(1- scores.mean()) 
        
    # Find where MSE is minimum
    minIndexMSE = MSE.index(min(MSE)) 
    
    # Find corresponding C value
    optimal_C = cValues[minIndexMSE]
    print ("The optimal value of C for training data is %d" % optimal_C)
    plt.plot(cValues, MSE)
    plt.xlabel('Parameter C Value')
    plt.ylabel('Misclassification Error')
    plt.show()
    
    # Return Optimal C value
    return optimal_C 