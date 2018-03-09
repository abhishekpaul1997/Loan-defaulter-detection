
# coding: utf-8

# In[169]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk


# In[6]:

df = pd.read_csv('C:/train.csv', na_values = ['#NAME?']) 


# In[7]:

df.head(40)


# In[8]:

print(df['DEFAULTER'].value_counts())


# In[227]:

X = df.drop('DEFAULTER', 1)
X = X.drop('LOAN_ID', 1)
Y = df.DEFAULTER


# In[228]:

X.head()


# In[229]:

X.isnull().sum().sort_values(ascending=False).head()


# In[230]:

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN' , strategy = 'median' , axis = 0)
imp.fit(X)
X = pd.DataFrame(data = imp.transform(X) , columns=X.columns)


# In[231]:

X.isnull().sum().sort_values(ascending=False).head()


# In[232]:

X.head(40)


# In[233]:

#Outlier Detection  using Turkey IQR
def find_outliers_turkey(x):
    q1 = np.percentile(x, 25)
    q3 = np.percentile(x, 75)
    iqr = q3 - q1
    floor = q1 - 1.5*iqr
    ceiling = q3 + 1.5*iqr
    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])
    outlier_values = list(x[outlier_indices])
    
    return outlier_indices, outlier_values     


# In[234]:

turkey_indices, turkey_values = find_outliers_turkey(X['AMOUNT'])
print(np.sort(turkey_values))


# In[235]:

def plot_histogram(x):
    plt.hist(x, color = 'grey', alpha = 0.5)
    plt.title("Histogram of {var_name}".format(var_name = x.name))
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.show()


# In[236]:

plot_histogram(X["AMOUNT"])


# In[237]:

def plot_histogram_dv(x, y):
    plt.hist(list(x[y==0]), alpha=0.5, label='DV=0')
    plt.hist(list(x[y==1]), alpha=0.5, label='DV=1')
    plt.title("Histogram of {var_name} by DV category".format(var_name = x.name))
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.legend(loc = 'upper right')
    plt.show()              


# In[238]:

plot_histogram_dv(X['AMOUNT'], Y)


# In[239]:

#Normalising features having signifant outliers
X['AMOUNT'] = np.log(X['AMOUNT'])
X['VALUE'] = np.log(X['VALUE'])
X['DUE_MORTGAGE'] = np.log(X['DUE_MORTGAGE'])


# In[240]:

#FEATURE ENGINEERING
from itertools import combinations
from sklearn.decomposition import PCA

pca = PCA(n_components=14)
X_pca = pd.DataFrame(pca.fit_transform(X))


# In[241]:

print(X_pca.head(5))


# In[242]:

#cross-validation
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, train_size=0.70, random_state=1)


# In[243]:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def find_model_perf(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    Y_hat = [x[1] for x in model.predict_proba(X_test)]
    auc = roc_auc_score(Y_test, Y_hat)
    
    return auc


# In[244]:

auc_processed = find_model_perf(LogisticRegression(), X_train, Y_train, X_test, Y_test)
print(auc_processed)


# In[245]:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# In[246]:

def classification_model(model, data, outcome):
    #Fit the model:
    model.fit(data,outcome)
  
    #Make predictions on training set:
    predictions = model.predict(data)
  
    #Print accuracy
    accuracy = metrics.accuracy_score(predictions,outcome)
    print ("Accuracy : %s" % "{0:.3%}".format(accuracy))

    #Perform k-fold cross-validation with 5 folds
    kf = KFold(data.shape[0], n_folds=10)
    error = []
    auc_error = []
    for train, test in kf:
    # Filter training data
        train_predictors = (data.iloc[train,:])
    
    # The target we're using to train the algorithm.
        train_target = outcome.iloc[train]
    
    # Training the algorithm using the predictors and target.
        model.fit(train_predictors, train_target)
    
    #Record error from each cross-validation run
        error.append(model.score(data.iloc[test,:], outcome.iloc[test]))
        auc_error.append(find_model_perf(model, data.iloc[train,:], outcome.iloc[train], data.iloc[test,:], outcome.iloc[test]))
    print ("Cross-Validation Score : %s" % "{0:.3%}".format(np.mean(error)))
    print ("AUC ROC : %s" % "{0:.3%}".format(np.mean(auc_error)))
    model.fit(data, outcome) 


# In[247]:

model = LogisticRegression()
classification_model(model, X_pca, Y)


# In[248]:

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
classification_model(model, X, Y)


# In[249]:

classification_model(DecisionTreeClassifier(), X, Y)


# In[250]:

from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

classification_model(model, X, Y)


# In[252]:

model = RandomForestClassifier(n_estimators=100)
classification_model(model, X_pca, Y)


# In[253]:

#Feature Engineering: Feature importance ranking
predictor_var=['AMOUNT','DUE_MORTGAGE','VALUE','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3']
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)


# In[254]:

model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
predictor_var = ['CONVICTED','VAR_1 ','CLT','OCC ','AMOUNT']
classification_model(model, X, Y)


# Classification Model using Ensembles

# In[257]:

#Using BaggingClassifier adn DecisionTreeClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

seed = 7
kfold = model_selection.KFold(n_splits=8, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
classification_model(model, X_pca, Y)


# In[262]:


from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=8, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees, max_features=max_features)
classification_model(model, X, Y)


# In[261]:

from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=8, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
classification_model(model, X, Y)


# In[260]:

from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=8, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
classification_model(model, X, Y)


# In[296]:

test = pd.read_csv('C:/test.csv', na_values = ['#NAME?']) 
test.head()


# In[297]:

#Imputing the test data with median values 
test_id = test['LOAN_ID']
test = test.drop('TEST_ID', 1)
test.head()


# In[298]:

test = test.drop('LOAN_ID', 1)
test.head()


# In[299]:

imp = Imputer(missing_values = 'NaN' , strategy = 'median' , axis = 0)
imp.fit(test)
test = pd.DataFrame(data = imp.transform(test) , columns=test.columns)


# In[300]:

test.head()


# In[330]:

#Uing GradientBooostingClassifier to train the entire training set
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=8, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
model.fit(X, Y)
prd = model.predict(test)


# In[349]:

output = pd.DataFrame(prd.view())


# In[335]:

test_id = pd.DataFrame(test_id)


# In[336]:

type(test_id)


# In[337]:

test_id.shape


# In[354]:

type(prd)


# In[353]:

output


# In[355]:

test_id['DEFAULTER']=prd


# In[356]:

test_id


# In[358]:

type(test_id)


# In[361]:

test_id.to_csv('D:/Submission.csv')


# In[ ]:



