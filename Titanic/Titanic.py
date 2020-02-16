# -*- coding: utf-8 -*-
"""
Column age has nan. It should be replaced by some values. For example: we can do it by two ways:
    1) try to find someone with the same last name and compare sibsp and parch values to understand is it child or not. If yes - nan should be replaced by average age of children.
    2) this is bad idea but it is a little bit easier. Find is it Mr or Ms or Mrs, after that replace nan by average age for this group 
    3) third idea is to use 1) and 2) together
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = r"D:\Kaggle\Titanic\titanic" + "\\"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "test.csv")

train.describe()
test.describe()
print(train.isnull().sum())
print(test.isnull().sum())

train.info()
test.info()
  
"""
Separating train into training and testing sets
"""
X = train.iloc[:, 0]
X = pd.concat([X, train.iloc[:, 2:]], axis = 1)
y_train = train.iloc[:, 1].values

def preprocessing(train):
    """
    Clearing 'Ticket' column
    """
    res = []
    for i in range(0,train['Ticket'].size):
        if [int(j) for j in train.loc[i,'Ticket'].split() if j.isdigit()] == []:
            res.append(None)
        else:
            res.append([int(j) for j in train.loc[i,'Ticket'].split() if j.isdigit()][0])
    train["clear_ticket"] = res
    
    """
    Clearing passenger's Status
    """
    status = []
    for i in range(0,train['Ticket'].size):
        status.append((train.loc[i,'Name'].split(", "))[1].split(".")[0])
    train["clear_status"] = status


    train = train.fillna(value = {'clear_ticket': 0, 'Embarked': 0})


    train = remove_nan_from_age(train, 3, 4, 5, 6, 7)
    train = train.fillna(train.mean())
    
    """
    Get dummies from Categorical variables
    """
    from sklearn.preprocessing import LabelEncoder
    import pandas as pd

    lbe = LabelEncoder()
    train['Sex'] = lbe.fit_transform(train['Sex'])
    dummies = pd.get_dummies(train.loc[:, ['Embarked', 'clear_status']], prefix = ['Embarked', 'Status'])
    train = pd.concat([train, dummies], axis = 1)
    train = train[['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
       'Parch', 'Fare', 'clear_ticket', 'Embarked_C', 'Embarked_Q', 'Status_Col', 'Status_Dr',
       'Status_Master', 'Status_Miss', 'Status_Mr', 'Status_Mrs',
       'Status_Ms', 'Status_Rev']]
    return train
    
    
 
X_train = preprocessing(X)
X_test = preprocessing(test)

print(X_train.isnull().sum())
print(X_test.isnull().sum())





"""
calculate mean within status values, but result is awfull. We can not use this to replace None in Age column
"""
status_mean = train_proc.groupby('clear_status')['Age'].mean().reset_index() 
fare_mean = test.fillna(test.mean())

"""
We can solve this problem by using clustering
Let's try to replace None in Age column by mediana of 11 next groups:
    1) SibSp == 0 & Parch == 0   ---- alone adult
    2) female & '(' in Name column & SibSp > 0 & Parch > 0  ---- adult, cause '(' is indicator of marriage
    3) female & '(' is not in Name column & Parch > 1  ---- child
    4) female & '(' is not in Name column & Parch == 1  ---- child
    5) female & '(' is not in Name column & SibSp > 0 & Parch == 0  ---- adult
    6) male &  SibSp == 0 & Parch == 1  ---- adult
    7) male &  SibSp == 0 & Parch > 1  ---- child
    8) male &  SibSp > 0 & Parch == 0  ---- adult
    9) male &  SibSp == 1 & Parch > 0  ---- adult
    10) male &  SibSp > 1 & Parch > 0  ---- child
    11) other
"""
def remove_nan_from_age(frame_orig, name_col_numb, sex_col_numb, Age_col_numb, sibsp_col_numb, parch_col_numb):
    frame = frame_orig
    cluster = []
    for i in range(0,frame['Ticket'].size):
        if frame.iloc[i, sibsp_col_numb] == 0 and frame.iloc[i, parch_col_numb] == 0:
            cluster.append(0)
        elif frame.iloc[i, sex_col_numb] == 'female' and '(' in frame.iloc[i, name_col_numb] and frame.iloc[i, sibsp_col_numb] > 0 and frame.iloc[i, parch_col_numb] > 0:
            cluster.append(1)
        elif frame.iloc[i, sex_col_numb] == 'female' and '(' not in frame.iloc[i, name_col_numb] and frame.iloc[i, parch_col_numb] > 1:
            cluster.append(2)
        elif frame.iloc[i, sex_col_numb] == 'female' and '(' not in frame.iloc[i, name_col_numb] and frame.iloc[i, parch_col_numb] == 1:
            cluster.append(3)
        elif frame.iloc[i, sex_col_numb] == 'female' and '(' not in frame.iloc[i, name_col_numb] and frame.iloc[i, parch_col_numb] == 0 and frame.iloc[i, sibsp_col_numb] > 0:
            cluster.append(4)
        elif frame.iloc[i, sex_col_numb] == 'male' and frame.iloc[i, sibsp_col_numb] == 0 and frame.iloc[i, parch_col_numb] == 1:
            cluster.append(5)
        elif frame.iloc[i, sex_col_numb] == 'male' and frame.iloc[i, sibsp_col_numb] == 0 and frame.iloc[i, parch_col_numb] > 1:
            cluster.append(6)
        elif frame.iloc[i, sex_col_numb] == 'male' and frame.iloc[i, sibsp_col_numb] > 0 and frame.iloc[i, parch_col_numb] == 0:
            cluster.append(7)
        elif frame.iloc[i, sex_col_numb] == 'male' and frame.iloc[i, sibsp_col_numb] == 1 and frame.iloc[i, parch_col_numb] > 0:
            cluster.append(8)
        elif frame.iloc[i, sex_col_numb] == 'male' and frame.iloc[i, sibsp_col_numb] > 1 and frame.iloc[i, parch_col_numb] > 0:
            cluster.append(9)
        else:
            cluster.append(10)
    frame['cluster'] = cluster
    cluster_median = frame.groupby('cluster')['Age'].median().reset_index() 
    for i in range(0,frame['Ticket'].size):
        if not frame.iloc[i,Age_col_numb] > -1:
            frame.iloc[i,Age_col_numb] = cluster_median.iloc[frame.iloc[i,frame.shape[1] - 1],1]
    return frame
            



"""
Split dataframe on test and train sets
"""
from sklearn.cross_validation import train_test_split
X_train_model, X_test_model, y_train_model, y_test_model = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)




"""
from sklearn.preprocessing import StandardScalersc_X = StandardScaler()X_train = sc_X.fit_transform(X_train)X_test = sc_X.transform(X_test)sc_y = StandardScaler()y_train = sc_y.fit_transform(y_train)
from sklearn.preprocessing import StandardScalersc_X = StandardScaler()X_train = sc_X.fit_transform(X_train)X_test = sc_X.transform(X_test)sc_y = StandardScaler()y_train = sc_y.fit_transform(y_train)
res = [int(i) for i in test_string.split() if i.isdigit()]


from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = "NaN", strategy = "mean", axis = 0)
imputer = imputer.fit(train.iloc[:, [5, 9]])
train.iloc[:, [5, 9]] = imputer.transform(train.iloc[:, [5, 9]])
"""



"""
Build a model
"""
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train_model, y_train_model)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test_model, classifier.predict(X_test_model))
print(cm)