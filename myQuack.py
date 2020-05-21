
'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.

'''
import time
import numpy as np
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


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ ( 10032029, 'Kaushal Kishorbhai', 'Limbasiya' ), (9954953, 'Lucas', 'Wickham'), (8890463, 'Michael', 'Gourlay') ]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset( dataset_path ):
    '''  
    Read a comma separated text file where 
	- the first field is a ID number 
	- the second field is a class label 'B' or 'M'
	- the remaining fields are real-valued

    Return two numpy arrays X and y where 
	- X is two dimensional. X[i,:] is the ith example
	- y is one dimensional. y[i] is the class label of X[i,:]
          y[i] should be set to 1 for 'M', and 0 for 'B'

    @param dataset_path: full path of the dataset text file

    @return
	X,y
    '''
    
    def get_size ( filename ):
        F = open ( filename, 'r' )
        line = F.readline ()
        array = line.split ( ',' )
        column_number = len ( array )
        with open ( filename ) as F:
                for i, l in enumerate ( F ):
                    pass
        return i + 1, column_number
    
    
    rows,columns = get_size ( dataset_path )
    F = open( dataset_path, 'r' )
    if ( F.mode == 'r' ):
        numberOfAttributes = columns - 2 #ignore ID and Y label columns for learners (first 2)
        numberOfObservations = rows
        #create empty numpy arrays
        X = np.zeros ( [ numberOfObservations, numberOfAttributes ] )
        y = np.zeros ( numberOfObservations )
        y = y.astype ( np.uint8 )
        #read each line and store it in the array
        for i in range ( 0, numberOfObservations ):
            contents = F.readline ()
            data = contents.strip ( '\n' )
            Array = data.split ( ',' )
            #if result is 'M' we set Y label to 1 otherwise leave it as 0
            if ( ( Array [ 1 ] ) == 'M' ):
                y [ i ] = 1
            Array = np.asarray ( Array [ 2: ] )
            X [ i ] = Array
            
        X = np.asarray ( X )
        y = np.asarray ( y )
        return X, y #return X and y
    
                
                
                
                
                
                
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_DecisionTree_classifier( X_training, y_training ):
    '''  
    Build a Decision Tree classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier( X_training, y_training, numberOfNeighbors ):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    
    knn_clf = KNeighborsClassifier( n_neighbors = numberOfNeighbors )
    knn_clf = knn_clf.fit( X_training, y_training )
    return knn_clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training, boxConstraint):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    
    svm_clf = svm.SVC( C = boxConstraint, gamma = 'scale', random_state = 8 )
    svm_clf = svm_clf.fit( X_training, y_training )
    return svm_clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NeuralNetwork_classifier(X_training, y_training):
    '''  
    Build a Neural Network classifier (with two dense hidden layers)  
    based on the training set X_training, y_training.
    Use the Keras functions from the Tensorflow library

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    
    #TESTING PARAMETERS
    iterations = 100 # max iterations on MLP
    
    
    
    # Load the dataset
    X , y = X_training, y_training
    
    # Split training set initially by %20
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size =(0.20), random_state =1 )
    
    # Preprocess data
    scaler = StandardScaler()
    scaler.fit( X_train )
    X_train_scaled = scaler.transform( X_train )
    X_test_scaled = scaler.transform( X_test )
    
    # Set a range of hidden layer values to cross-validate over
    hidden_layers_list = [ ( 10, 10 ), ( 30, 20 ), ( 50, 50 ), ( 20, 20 ) ]
    mlp = MLPClassifier( hidden_layer_sizes = hidden_layers_list,max_iter=iterations, verbose = True, random_state = 1 )
    #Set parameters list to grid seach 
    parameters = { 'hidden_layer_sizes':hidden_layers_list }
    clf = GridSearchCV( mlp, parameters, scoring = "accuracy" )
   
    #Fit to training data
    clf.fit( X_train_scaled, y_train )
    
    #Get predictions from model and ouput reports
    y_pred = clf.predict( X_test_scaled )
    matrix = confusion_matrix( y_test, y_pred )
    best_params = clf.best_params_
    tp,fn,fp,tn = matrix.ravel()
    
    #Print Results
    print( "\nConfusion Matrix \n", matrix )
    print( "Classification Report \n", classification_report( y_test, y_pred ) )
    print( "Best Hidden Layer Parmeter", best_params )
    print( "True Positve\n",tp,  " \nFalse Negative\n", fn, "\nFalse Positive\n", fp, "\nTrue Negative\n", tn )
    print( "Total Accuracy", ( tn + tp ) * 100 / ( tp + tn + fp + fn ) )
    print( "Total Precision", tp * 100 / ( tp + fp ) )
  
   
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ## AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
    ##         "INSERT YOUR CODE HERE"  
    
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
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        
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
        knn = build_NearrestNeighbours_classifier( X_training, y_training, numberOfNeighbors = k ) 
        
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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

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

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
def NearestNeighbours():
    '''
    Performs tests for Nearest Neighbors Classifier to evaluate best number of neighbours
    and report errors and accuracy on it.
    '''
    print ( "------------------------------------------------------" )
    print ( "------------------------------------------------------" )
    print ( "Training Data Graph for KNN Classifier" )
    optimalNumberOfNeighbours = knnTests ( X_training, y_training, neighbours )
    print ( "Accuracy on Training Data with optimal number of neighbours: " + str ( knnErrors ( X_training, y_training, optimalNumberOfNeighbours ) * 100) + "%" )
    print ( "Accuracy on Test Data with optimal number of neighbours: " + str ( knnErrors ( X_testing, y_testing, optimalNumberOfNeighbours) * 100 ) + "%" )
    print ( "Accuracy on Validation Data with optimal number of neighbors: " + str(knnErrors(X_validation, y_validation, optimalNumberOfNeighbours ) * 100) + "%" )
    print ( "------------------------------------------------------" )
    print ( "------------------------------------------------------" )
    print ( "Error on Training Data with optimal number of neighbors: " + str ( 100 - knnErrors ( X_training, y_training, optimalNumberOfNeighbours ) * 100 ) + "%" )
    print ( "Error on Test Data with optimal number of neighbors: " + str ( 100 - knnErrors ( X_testing, y_testing, optimalNumberOfNeighbours ) * 100 ) + "%" )
    print ( "Error on Validation Data with optimal number of neighbors: " + str ( 100 - knnErrors ( X_validation, y_validation, optimalNumberOfNeighbours ) * 100 ) + "%" )
    print ( "" )
    print ( "" )
    print ( "" )
        
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    
def SVM():
    '''
    Performs tests for SVM Classifier to evaluate best value of C
    and report errors and accuracy on it.
    '''
    print ( "------------------------------------------------------" )
    print ( "------------------------------------------------------" )
    print ( "Training Data Graph for SVM CLassifier" )
    boxConstraint = svmTests( X_training, y_training, cValues )
    print ( "Accuracy on Training Data with optimal value of C: " + str ( svmErrors ( X_training, y_training, boxConstraint ) * 100 ) + "%" )
    print ( "Accuracy on Test Data with optimal value of C: " + str ( svmErrors ( X_testing, y_testing, boxConstraint ) * 100 ) + "%" )
    print ( "Accuracy on Validation Data with optimal value of C: " + str ( svmErrors ( X_validation, y_validation, boxConstraint ) * 100 ) + "%" )
    print ( "------------------------------------------------------" )
    print ( "------------------------------------------------------" )
    print ( "Error on Training Data with optimal value of C: " + str ( 100 - svmErrors ( X_training, y_training, boxConstraint ) * 100 ) + "%" )
    print ( "Error on Test Data with optimal value of C: " + str ( 100 - svmErrors ( X_testing, y_testing, boxConstraint ) * 100 ) + "%" )
    print ( "Error on Validation Data with optimal value of C: " + str ( 100 - svmErrors ( X_validation, y_validation, boxConstraint ) * 100 ) + "%" )
    print ( "" )
    print ( "" )
    print ( "" )

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    ##         "INSERT YOUR CODE HERE"    
    print( my_team() )
    X, y = prepare_dataset( 'medical_records.data' )
    X_training, X_testing, y_training, y_testing = train_test_split ( X, y, testing_size = 0.2, shuffle = False, random_state = 1 )
    X_training, X_validation, y_training, y_validation = train_test_split ( X_training, y_training, testing_size = 0.2, shuffle = False, random_state = 1 )
    
    
    # Records considered for each dataset
    
    print ( "Number of records for training : " + str ( X_training.shape [ 0 ] ) )
    print()
    print()
    print ("Number of records for testing : " + str ( X_testing.shape [ 0 ] ) )
    print()
    print()
    print ("Number of records for validation : " + str ( X_validation.shape [ 0 ] ) )
    print()
    print()
    
    
    # Neighbours array for KNN Classifier
    neighbours = [ 1, 3, 5, 7, 9, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 43, 55 ] 
    
    # C Values for SVM CLassifier
    cValues = np.arange ( 1, 50, 0.5 ) 
    
    NearestNeighbours()
    SVM()
    
    build_NeuralNetwork_classifier( X, y )
 


