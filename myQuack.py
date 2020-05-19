
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
from sklearn.metrics import confusion_matrix, classification_report


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

def build_NearrestNeighbours_classifier(X_training, y_training):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_SupportVectorMachine_classifier(X_training, y_training):
    '''  
    Build a Support Vector Machine classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    '''
    ##         "INSERT YOUR CODE HERE"    
    raise NotImplementedError()

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
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =(0.20), random_state =1)
    
    # Preprocess data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Set a range of hidden layer values to cross-validate over
    hidden_layers_list = [(10,10),(30,20),(50,50),(20,20)]
    mlp = MLPClassifier(hidden_layer_sizes = hidden_layers_list,max_iter=iterations, verbose = True, random_state = 1)
    #Set parameters list to grid seach 
    parameters = {'hidden_layer_sizes':hidden_layers_list}
    clf = GridSearchCV(mlp, parameters, scoring = "accuracy")
   
    #Fit to training data
    clf.fit(X_train_scaled,y_train)
    
    #Get predictions from model and ouput reports
    y_pred = clf.predict(X_test_scaled)
    matrix = confusion_matrix(y_test, y_pred)
    best_params = clf.best_params_
    tp,fn,fp,tn = matrix.ravel()
    
    #Print Results
    print("\nConfusion Matrix \n",matrix)
    print("Classification Report \n", classification_report(y_test, y_pred))
    print("Best Hidden Layer Parmeter", best_params)
    print("True Positve\n",tp,  " \nFalse Negative\n",fn,"\nFalse Positive\n", fp,"\nTrue Negative\n",tn)
    print("Total Accuracy",(tn+tp)*100/(tp+tn+fp+fn))
    print("Total Precision", tp*100/(tp+fp))
  
   
    

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    ## AND OTHER FUNCTIONS TO COMPLETE THE EXPERIMENTS
    ##         "INSERT YOUR CODE HERE"    
  

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    pass
    # Write a main part that calls the different 
    # functions to perform the required tasks and repeat your experiments.
    # Call your functions here

    ##         "INSERT YOUR CODE HERE"    
    #print(my_team())
    X, y = prepare_dataset('medical_records.data')
    build_NeuralNetwork_classifier(X,y)
 


