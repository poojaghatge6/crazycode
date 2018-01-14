import numpy as np
import pandas as pd
import time

np.random.seed(int(time.time())) 

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')


full = train.append(test, ignore_index = True)
titanic = full[ :891]
#print ('Datasets:' , 'full:' , full.shape , 'titanic:' , titanic.shape)


#Pre-processing
sex = pd.Series( np.where( full.Sex == 'male' , 1 , 0 ) , name = 'Sex' )
embarked = pd.get_dummies( full.Embarked , prefix='Embarked' )
pclass = pd.get_dummies( full.Pclass , prefix='Pclass' )
# Create dataset
imputed = pd.DataFrame()


imputed[ 'Age' ] = full.Age.fillna( full.Age.mean() )


# Fill missing values of Fare with the average of Fare (mean)
imputed[ 'Fare' ] = full.Fare.fillna( full.Fare.mean() )


title = pd.DataFrame()
# we extract the title from each name
title[ 'Title' ] = full[ 'Name' ].map( lambda name: name.split( ',' )[1].split( '.' )[0].strip() )


# a map of more aggregated titles
Title_Dictionary = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }


# we map each title
title[ 'Title' ] = title.Title.map( Title_Dictionary )
title = pd.get_dummies( title.Title )


cabin = pd.DataFrame()
# replacing missing cabins with U (for Uknown)
cabin[ 'Cabin' ] = full.Cabin.fillna( 'U' )
# mapping each Cabin value with the cabin letter
cabin[ 'Cabin' ] = cabin[ 'Cabin' ].map( lambda c : c[0] )
# dummy encoding
cabin = pd.get_dummies( cabin['Cabin'] , prefix = 'Cabin' )

    

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else:
        return 'XXX'


ticket = pd.DataFrame()
# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )


ticket.shape
ticket.head()


family = pd.DataFrame()


# introducing a new feature : the size of families (including the passenger)
family[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1


# introducing other features based on the family size
family[ 'Family_Single' ] = family[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
family[ 'Family_Small' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
family[ 'Family_Large' ]  = family[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )




# Select which features/variables to include in the dataset from the list below:
# imputed , embarked , pclass , sex , family , cabin , ticket
full_X = pd.concat( [ imputed , embarked , pclass, cabin , sex, family] , axis=1 )
full_X.head()


#Create dataset - split data to training set and test set
# Create all datasets that are necessary to train, validate and test models
train_valid_X = full_X[ 0:891 ]
train_valid_y = titanic.Survived
test_X = full_X[ 891: ]


result_set = pd.read_csv('gender_submission.csv')
y_test = result_set.iloc[ :, 1].values


#Use Support Vector Machines model to train data set and predict data
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(train_valid_X, train_valid_y)


predictions = classifier.predict(train_valid_X)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(train_valid_y,predictions)


#Generate confusion matrix and print out prediction in csv format
test_predictions = classifier.predict(test_X)
test_cm = confusion_matrix(y_test,test_predictions)
#print("PassengerId,Survived")
#for passengerId in test.iloc[:,0]:
#    print(str(passengerId)+','+str(test_predictions[passengerId-892]))

test_X = np.transpose(test_X)
#print(test_X.shape)
#print(train_valid_X)
train_valid_X = np.transpose(train_valid_X).values

#train_valid_X = train_valid_X[1:]
#print(train_valid_X)
train_valid_Y = np.transpose(train_valid_y)
#print(train_valid_Y.shape)
tr_y = []

for i in train_valid_Y:
    tr_y.append(i)
    
#print (len(tr_y))
    
train_y = pd.DataFrame(np.array(tr_y).reshape(1,891)).values

#print(train_y)
#print(train_valid_y.shape)
#print(len(test_X)) 
n_x = 22
n_h = 10
n_y = 1

def sigmoid(z):
    s = 1/(1+np.exp(-z))
    return s

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(int(time.time())) 
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))   
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}   
    return parameters
             

def forward_propagation(X, parameters):    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]    
    Z1 = np.dot(W1,X) + b1               
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    cache = {"Z1": Z1, "A1": A1,"Z2": Z2,"A2": A2}
    return A2, cache


def compute_cost(A2, Y, parameters):
    m = Y.shape[1] 
    logprobs = np.multiply(np.log(A2),Y)+np.multiply(np.log(1-A2),1-Y)
    cost = - np.sum(logprobs)/m   
    cost = np.squeeze(cost)
    return cost



def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']
    dZ2 = A2-Y
    dW2 = (np.dot(dZ2,A1.T))/m
    db2 = np.sum(dZ2,axis=1,keepdims=True)/m
    dZ1 = np.dot(W2.T,dZ2)*(1 - np.power(A1, 2))
    dW1 = np.dot(dZ1,X.T)/m
    db1 = np.sum(dZ1,axis=1,keepdims=True)/m
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    return grads

def update_parameters(parameters, grads, learning_rate = 0.01):
    W1 = parameters["W1"]  
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

def nn_model(X, Y, n_h, num_iterations=10000, print_cost=True):
    np.random.seed(3)
    n_x = 22
    n_y = 1
    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    for i in range(0, num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y, parameters)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads)
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    return parameters



parameters = nn_model(train_valid_X, train_y, 10, num_iterations=500000, print_cost=True)

# GRADED FUNCTION: predict

def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions
np.set_printoptions(threshold=np.nan)
#param = initialize_parameters()
#print (y_test)
predictions = predict(parameters, test_X)
#print (predictions)

for i in predictions:
    print(i)

f = open("output1",'w')
f.write(str(predictions))
f.close()
#d = model(train_valid_X, train_y, test_X, y_test, num_iterations = 2000, learning_rate = 0.005, print_cost = True)
#print("W1 = " + str(parameters["W1"]))
#print("b1 = " + str(parameters["b1"]))
#print("W2 = " + str(parameters["W2"]))
#print("b2 = " + str(parameters["b2"]))
#df = pd.DataFrame()
#for i in range(0,891):
#    df.append(pd.DataFrame(train_valid_X[i]))
#print (df)
#print (train_y)