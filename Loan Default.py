
# coding: utf-8

# In[1177]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk


# In[1178]:

df = pd.read_csv('C:/train.csv', na_values = ['#NAME?']) 


# In[1179]:

df.head(40)


# In[1180]:

print(df['DEFAULTER'].value_counts())


# In[1181]:

X = df.drop('DEFAULTER', 1)
X = X.drop('LOAN_ID', 1)
Y = df.DEFAULTER


# In[1182]:

X.head()


# In[1183]:

X.isnull().sum().sort_values(ascending=False).head()


# In[1184]:

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values = 'NaN' , strategy = 'median' , axis = 0)
imp.fit(X)
X = pd.DataFrame(data = imp.transform(X) , columns=X.columns)


# In[1185]:

X.isnull().sum().sort_values(ascending=False).head()


# In[1186]:

X.head(40)


# In[1187]:

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


# In[1188]:

turkey_indices, turkey_values = find_outliers_turkey(X['AMOUNT'])
print(np.sort(turkey_values))


# In[1189]:

def plot_histogram(x):
    plt.hist(x, color = 'grey', alpha = 0.5)
    plt.title("Histogram of {var_name}".format(var_name = x.name))
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.show()


# In[1190]:

plot_histogram(X["AMOUNT"])


# In[1191]:

def plot_histogram_dv(x, y):
    plt.hist(list(x[y==0]), alpha=0.5, label='DV=0')
    plt.hist(list(x[y==1]), alpha=0.5, label='DV=1')
    plt.title("Histogram of {var_name} by DV category".format(var_name = x.name))
    plt.xlabel("value")
    plt.ylabel("Frequency")
    plt.legend(loc = 'upper right')
    plt.show()              


# In[1192]:

plot_histogram_dv(X['AMOUNT'], Y)


# In[1193]:

#Normalising features having signifant outliers
X['AMOUNT'] = np.log(X['AMOUNT'])
X['VALUE'] = np.log(X['VALUE'])
X['DUE_MORTGAGE'] = np.log(X['DUE_MORTGAGE'])


# In[1194]:

#FEATURE ENGINEERING
from itertools import combinations
from sklearn.decomposition import PCA

pca = PCA(n_components=14)
X_pca = pd.DataFrame(pca.fit_transform(X))


# In[1195]:

print(X_pca.head(5))


# In[1196]:

#cross-validation
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, train_size=0.70, random_state=1)


# In[1197]:

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def find_model_perf(model, X_train, Y_train, X_test, Y_test):
    model.fit(X_train, Y_train)
    Y_hat = [x[1] for x in model.predict_proba(X_test)]
    auc = roc_auc_score(Y_test, Y_hat)
    
    return auc


# In[1198]:

auc_processed = find_model_perf(LogisticRegression(), X_train, Y_train, X_test, Y_test)
print(auc_processed)


# In[1199]:

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold   #For K-fold cross validation
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics


# In[1200]:

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


# In[1201]:

model = LogisticRegression()
classification_model(model, X_pca, Y)


# In[1202]:

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
classification_model(model, X, Y)


# In[1203]:

classification_model(DecisionTreeClassifier(), X, Y)


# In[1204]:

from sklearn.ensemble import GradientBoostingClassifier
model= GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=0)

classification_model(model, X, Y)


# In[1205]:

#Feature Engineering: Feature importance ranking
predictor_var=['AMOUNT','DUE_MORTGAGE','VALUE','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3']
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)


# In[1206]:

model = RandomForestClassifier(n_estimators=100)
classification_model(model, X_pca, Y)


# In[1207]:

#Feature Engineering: Feature importance ranking
predictor_var=['AMOUNT','DUE_MORTGAGE','VALUE','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3']
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)


# In[1208]:

model = RandomForestClassifier(n_estimators=25, min_samples_split=25, max_depth=7, max_features=1)
reduced_features = ['DCL','AMOUNT','CLT','REASON','CL_COUNT','DUE_MORTGAGE']
X_new=X[reduced_features]
classification_model(model, X, Y)
predictor_var=['AMOUNT','DUE_MORTGAGE','VALUE','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3']
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)


# Classification Model using Ensembles

# In[1209]:

#Using BaggingClassifier adn DecisionTreeClassifier
from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

seed = 7
kfold = model_selection.KFold(n_splits=8, random_state=seed)
cart = DecisionTreeClassifier()
num_trees = 100
model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=seed)
classification_model(model, X, Y)


# In[1211]:


from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier

seed = 7
num_trees = 100
max_features = 3
kfold = model_selection.KFold(n_splits=8, random_state=seed)
model = RandomForestClassifier(n_estimators=num_trees ,random_state=0)
classification_model(model, X, Y)
predictor_var=['AMOUNT','DUE_MORTGAGE','VALUE','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3']
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)


# In[1212]:

from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier

seed = 7
num_trees = 30
kfold = model_selection.KFold(n_splits=8, random_state=seed)
model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)
classification_model(model, X, Y)
predictor_var=['AMOUNT','DUE_MORTGAGE','VALUE','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3']
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)


# In[1213]:

from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier

seed = 7
num_trees = 100
kfold = model_selection.KFold(n_splits=8, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
classification_model(model, X, Y)
predictor_var=['AMOUNT','DUE_MORTGAGE','VALUE','REASON','OCC','TJOB','DCL','CLT','CL_COUNT','RATIO','CONVICTED','VAR_1','VAR_2','VAR_3']
featimp = pd.Series(model.feature_importances_, index=predictor_var).sort_values(ascending=False)
print (featimp)


# In[1214]:

test = pd.read_csv('C:/test.csv', na_values = ['#NAME?']) 
test.head()


# In[1215]:

#Imputing the test data with median values 
test_id = test['LOAN_ID']
test = test.drop('TEST_ID', 1)
test.head()


# In[1216]:

test = test.drop('LOAN_ID', 1)
test.head()


# In[1217]:

imp = Imputer(missing_values = 'NaN' , strategy = 'median' , axis = 0)
imp.fit(test)
test = pd.DataFrame(data = imp.transform(test) , columns=test.columns)


# In[1218]:

test.head()


# In[1219]:

#Uing GradientBooostingClassifier to train the entire training set
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier
reduced_features = ['RATIO','CLT','VALUE','CL_COUNT','DUE_MORTGAGE']
X_new=X[reduced_features]
test_new=test[reduced_features]
seed = 9
num_trees = 14
kfold = model_selection.KFold(n_splits=200, random_state=seed)
model = GradientBoostingClassifier(n_estimators=num_trees, random_state=seed)
classification_model(model, X_new, Y)
model.fit(X_new, Y)
prd = model.predict(test_new)



# In[1220]:

model.fit(X_new, Y)
prd = model.predict(test_new)


# In[1221]:

output = pd.DataFrame(prd.view())


# In[1222]:

test_id = pd.DataFrame(test_id)


# In[1223]:

type(test_id)


# In[1224]:

test_id.shape


# In[1225]:

type(prd)


# In[1226]:

output   


# In[1227]:

test_id['DEFAULTER']=prd
sum(test_id['DEFAULTER'])


# In[1228]:

test_id


# In[1229]:

type(test_id)


# In[1232]:

test_id.to_csv('D:/Submission.csv')


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



