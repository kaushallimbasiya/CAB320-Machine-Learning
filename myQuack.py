
'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.

'''

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import csv
import matplotlib.pyplot as plt
from sklearn import metrics


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def my_team():
    '''
    Return the list of the team members of this assignment submission as a list
    of triplet of the form (student_number, first_name, last_name)
    
    '''
    return [ ( 10032029, 'Kaushal Kishorbhai', 'Limbasiya' ), (9954953, 'Lucas', 'Wickham'), (8890463, 'Michael', 'Gourlay') ]


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def prepare_dataset(dataset_path):
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

def build_DecisionTree_classifier(X_training, y_training):
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

def build_NearrestNeighbours_classifier(X_training, y_training, numberOfNeighbors):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"
    
    knn_clf = KNeighborsClassifier(n_neighbors = numberOfNeighbors)
    knn_clf = knn_clf.fit(X_training, y_training)
    return knn_clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training, box_Constraint):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    
    svm_clf = svm.SVC(C = box_Constraint, gamma = 'scale', random_state = 8)
    svm_clf = svm_clf.fit(X_training, y_training)
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
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ## AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    ##         "INSERT YOUR CODE HERE"    
    my_team()
    #prepare_dataset()
    


