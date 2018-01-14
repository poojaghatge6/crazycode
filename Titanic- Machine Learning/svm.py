#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 15:37:40 2017

@author: chaoguo
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score

#import dataset
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

#combine test set with train set
full = train.append(test, ignore_index = True)
titanic = full[ :891]

sex = pd.Series( np.where(full.Sex=='male', 1, 0), name='Sex')
embarked = pd.get_dummies(full.Embarked, prefix='Embarked')
pclass = pd.get_dummies(full.Pclass, prefix='Pclass')

# Create dataset
imputed = pd.DataFrame()
#fill missing value with the average value
imputed['Age'] = full.Age.fillna(full.Age.mean())
imputed['Fare'] = full.Fare.fillna(full.Fare.mean())

title = pd.DataFrame()
#extract the title from name
title['Title'] = full['Name'].map(lambda name: name.split(',')[1].split('.')[0].strip())

title_dict = {
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

#map each title
title['Title'] = title.Title.map(title_dict)
title = pd.get_dummies(title.Title)

cabin = pd.DataFrame()
# replacing missing value
cabin['Cabin'] = full.Cabin.fillna('U')
# mapping each Cabin value with the cabin letter
cabin['Cabin'] = cabin['Cabin'].map(lambda c: c[0])
cabin = pd.get_dummies(cabin['Cabin'], prefix='Cabin')

# a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
def cleanTicket(ticket):
    ticket = ticket.replace('.', '')
    ticket = ticket.replace('/', '')
    ticket = ticket.split()
    ticket = map(lambda t: t.strip(), ticket)
    ticket = list(filter(lambda t: not t.isdigit(), ticket))
    if len(ticket) > 0:
        return ticket[0]
    else:
        return 'XXX'

ticket = pd.DataFrame()
# Extracting dummy variables from tickets:
ticket['Ticket'] = full['Ticket'].map(cleanTicket)
ticket = pd.get_dummies(ticket['Ticket'], prefix='Ticket')

family = pd.DataFrame()
# introducing a new feature : the size of families (including the passenger)
family['FamilySize'] = full['Parch'] + full['SibSp'] + 1

# introducing other features based on the family size
family['Family_Single'] = family['FamilySize'].map(lambda s: 1 if s == 1 else 0)
family['Family_Small'] = family['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
family['Family_Large'] = family['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

# Select features to be included in the dataset
full_X = pd.concat([imputed, embarked, pclass, cabin, sex, family], axis=1)

# Create all datasets that are necessary to train, validate and test models
X_train = full_X[0:891]
y_train = titanic.Survived
X_test = full_X[891:]

#define a score function to compute accuracy score
def compute_score(clf, X, y, scoring='accuracy'):
    xval = cross_val_score(clf, X, y, cv = 5, scoring=scoring)
    return np.mean(xval)

from sklearn.svm import SVC
classifier = SVC(kernel='linear', random_state=0)

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)
#generate output file
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
submission.to_csv('output.csv', index=False)
