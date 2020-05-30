
'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.

'''
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
        
        #Scale X_training Data
        scaler = StandardScaler()
        scaler.fit(X)
        X_train = scaler.transform(X)
        
        return X_train, y #return X and y        
    
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

    clf_unrestrained = DecisionTreeClassifier(random_state=0)
    clf_unrestrained.fit(X_training, y_training)
    depth_max = clf_unrestrained.get_depth()

    params = {'max_depth': list(range(1,depth_max))}

    clf = GridSearchCV(DecisionTreeClassifier(), params, scoring='f1_macro', cv=4)
    clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def build_NearrestNeighbours_classifier( X_training, y_training ):
    '''  
    Build a Nearrest Neighbours classifier based on the training set X_training, y_training.

    @param 
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]

    @return
	clf : the classifier built in this function
    ''' 
    
    params = {'n_neighbors': [ 1, 3, 5, 7, 9, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 43, 55 ] } 

    clf = GridSearchCV(KNeighborsClassifier(), params, scoring='f1_macro', cv=4)
    clf.fit( X_training, y_training )
    return clf

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
    
    params = {'C': list(np.arange( 1, 50, 0.5 ))}

    svm_clf = svm.SVC(gamma = 'scale', random_state = 0)
    clf = GridSearchCV(svm_clf, params, scoring='f1_macro', cv=4)
    clf.fit(X_training, y_training) 
    return clf

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
    
    #TESTING PARAMETERS
    iterations = 150 # max iterations on MLP
        
    # Set a range of hidden layer values to cross-validate over
    hidden_layers_list = [ ( 10, 10 ), ( 30, 20 ), ( 50, 50 ), ( 20, 20 ) ]
    
    
    mlp = MLPClassifier( hidden_layer_sizes = hidden_layers_list,max_iter=iterations, verbose = False, random_state = 1 )
    
    #Set parameters list to grid seach 
    params = { 'hidden_layer_sizes':hidden_layers_list }
    clf = GridSearchCV( mlp, params, scoring = 'f1_macro', cv=4 , n_jobs= -1)
   
    #Fit to training data
    clf.fit( X_training, y_training )
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def split_dataset(X, y):
    '''
    Splits X and y into training, validation and testing sets in tuple form
    '''
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 0.2, shuffle = False, random_state = 1)
    X_training, X_validation, y_training, y_validation = train_test_split(X_training, y_training, test_size = 0.2, shuffle = False, random_state = 1)
    return ((X_training, y_training), (X_validation, y_validation),(X_testing, y_testing))

def evaluate_classifier_test(classifier, X_testing, y_testing):
    '''
    Evaluates classifier on test data set
    '''
    y_pred = classifier['clf'].predict(X_testing)
    score = metrics.accuracy_score(y_testing, y_pred)
    print("Testing accuracy on {}:\t{}\n".format(classifier['name'], score))

def report(classifiers, datasets):
    '''
    Generates a report comparing the classifiers
    '''
    datasets_str = '\t\t\t'
    for d in datasets:
        datasets_str += d['name'] + '\t\t'
    print(datasets_str)
    for c in classifiers:
        clf_str = c['name'] + '\t'
        for d in datasets:
            y_pred = c['clf'].predict(d['X'])
            score = metrics.accuracy_score(d['y'], y_pred)
            clf_str += '{}\t'.format(score)
        print(clf_str + '\n')

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

if __name__ == "__main__":
    print(my_team())

    # prepare data
    X, y = prepare_dataset('medical_records.data')

    ((X_training, y_training), (X_validation, y_validation),(X_testing, y_testing)) = split_dataset(X, y)

    print ("Number of records for training : " + str(X_training.shape[0]))
    print ("Number of records for testing : " + str(X_testing.shape[0]))
    print ("Number of records for validation : " + str(X_validation.shape[0]))

    datasets = [
        {"name": "Training", "X": X_training, "y": y_training},
        {"name": "Validation", "X": X_validation, "y": y_validation},
        {"name": "Testing", "X": X_testing, "y": y_testing},
    ]

    # classifiers
    decision_tree_clf = build_DecisionTree_classifier(X_training, y_training)
    nearest_neighbours_clf = build_DecisionTree_classifier(X_training, y_training)
    support_vector_machine_clf = build_SupportVectorMachine_classifier(X_training, y_training)
    neural_network_clf = build_NeuralNetwork_classifier(X_training, y_training)

    classifiers = [
        {"name": "Decision Tree         ", "clf": decision_tree_clf},
        {"name": "Nearest Neighbours    ", "clf": nearest_neighbours_clf},
        {"name": "Support Vector Machine", "clf": support_vector_machine_clf},
        {"name": "Neural Network        ", "clf": neural_network_clf},
    ]

    # test performance
    for classifier in classifiers:
        evaluate_classifier_test(classifier, X_testing, y_testing)

    # generate report
    report(classifiers, datasets)
