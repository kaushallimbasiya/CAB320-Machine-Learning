
'''

2020 

Scaffolding code for the Machine Learning assignment. 

You should complete the provided functions and add more functions and classes as necessary.
 
You are strongly encourage to use functions of the numpy, sklearn and tensorflow libraries.

You are welcome to use the pandas library if you know it.

'''

import csv
import time
import numpy as np
import sklearn as sk
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

cv_num = 3 # cross validation number

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
    feature_cols = list(range(2,32)) # features are on cols index 2 to 31
    X = np.loadtxt(dataset_path,delimiter=',',dtype=float,usecols=feature_cols)
    y = np.loadtxt(dataset_path,delimiter=',',dtype=str,usecols=1)

    # scale X
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)

    # convert y to int
    y[y == 'B'] = 0
    y[y == 'M'] = 1
    y = y.astype(int)

    return X, y     
    
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
    # use unrestrained clf to get upper bound of max_depth
    clf_unrestrained = DecisionTreeClassifier(random_state=0)
    clf_unrestrained.fit(X_training, y_training)
    depth_max = clf_unrestrained.get_depth()

    params = {'max_depth': list(range(1,depth_max))}

    clf = GridSearchCV(DecisionTreeClassifier(), params, scoring='f1', cv=cv_num)
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
    params = {'n_neighbors': range(1, 100) } 

    clf = GridSearchCV(KNeighborsClassifier(), params, scoring='f1', cv=cv_num)
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

    clf = GridSearchCV(svm_clf, params, scoring='f1', cv=cv_num)
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
    # Set a range of hidden layer values to cross-validate over
    hidden_layers_list = [ ( 10, 10 ), ( 30, 20 ), ( 50, 50 ), ( 20, 20 ) ]
    params = { 'hidden_layer_sizes': hidden_layers_list }

    # alt params
    #params = {'hidden_layer_sizes': range(1, 100, 5)}

    mlp_clf = MLPClassifier(max_iter=150, verbose = False, random_state = 0, tol=0.001)

    clf = GridSearchCV(mlp_clf, params, scoring = 'f1', cv=cv_num , n_jobs= -1)

    #Fit to training data
    clf.fit(X_training, y_training)
    return clf

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def split_dataset(X, y):
    '''
    Splits X and y into training, validation and testing sets in tuple form

    @param 
	X: X[i,:] is the ith example
	y: y[i] is the class label of X[i,:]
    '''
    # split data into training and test
    X_training, X_testing, y_training, y_testing = train_test_split(X, y, test_size = 0.2, shuffle = False, random_state = 1)
    
    # further split training data into training and validation
    X_training, X_validation, y_training, y_validation = train_test_split(X_training, y_training, test_size = 0.2, shuffle = False, random_state = 1)
    
    return ((X_training, y_training), (X_validation, y_validation), (X_testing, y_testing))

def evaluate_classifier_test(classifier, X_testing, y_testing):
    '''
    Evaluates classifier on test data set

    @param 
    classifier: dict with classifier name and clf
	X_training: X_training[i,:] is the ith example
	y_training: y_training[i] is the class label of X_training[i,:]
    '''
    # predict on test data
    y_pred = classifier['clf'].predict(X_testing)

    # score on the expected result
    score = metrics.accuracy_score(y_testing, y_pred)

    print("Testing accuracy on {}:\t{}\n".format(classifier['name'], score))

def report(classifiers, datasets):
    '''
    Generates a report comparing the classifiers

    @param 
    classifiers: list of dict with classifier name and clf
    datasets: list of dict with dataset name and clf
    '''
    # header with dataset names
    datasets_str = '\t\t\t'
    for d in datasets:
        datasets_str += d['name'] + '\t\t'
    print(datasets_str)

    # print the results for each classifier
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

    ((X_training, y_training), (X_validation, y_validation), (X_testing, y_testing)) = split_dataset(X, y)

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
