import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


data=pd.read_csv('Churn_Modelling.csv')
print data.head()
'''
   RowNumber  CustomerId   Surname  ...  IsActiveMember EstimatedSalary Exited
0          1    15634602  Hargrave  ...               1       101348.88      1
1          2    15647311      Hill  ...               1       112542.58      0
2          3    15619304      Onio  ...               0       113931.57      1
3          4    15701354      Boni  ...               0        93826.63      0
4          5    15737888  Mitchell  ...               1        79084.10      0

[5 rows x 14 columns]'''

print data.shape   # (10000, 14)

# Finding out the correlation using graph

# get correlation of each feature in dataset
corrmat = data.corr()
top_corr_features = corrmat.index
plt.figure(figsize =(20,20))
g=sns.heatmap(data[top_corr_features].corr(),annot=True,cmap='RdYlGn')
plt.savefig('Correlation_map_of_each_feature')
plt.show()

# Devide dependent and independent variables

x = data.iloc[:,3:13]
y = data.iloc[:,13]

print y[:6]
'''
0    1
1    0
2    1
3    0
4    0
5    1
Name: Exited, dtype: int64'''

print data.isnull().sum()
'''
RowNumber          0
CustomerId         0
Surname            0
CreditScore        0
Geography          0
Gender             0
Age                0
Tenure             0
Balance            0
NumOfProducts      0
HasCrCard          0
IsActiveMember     0
EstimatedSalary    0
Exited             0
dtype: int64'''


print data['Geography'].unique()  # ['France' 'Spain' 'Germany']
print data['Geography'].value_counts()

'''
France     5014
Germany    2509
Spain      2477
Name: Geography, dtype: int64'''

geography = pd.get_dummies(x['Geography'],drop_first=True)
print geography.head()
'''
   Germany  Spain
0        0      0
1        0      1
2        0      0
3        0      0
4        0      1'''

gender =  pd.get_dummies(x['Gender'],drop_first=True)
print gender.head()
'''
   Male
0     0
1     0
2     0
3     0
4     0'''

# we have to drop those dummy columns from our dataset

x=x.drop(['Geography','Gender'],axis =1)
print x.head()

# concatenate those columns

x=pd.concat([x,geography,gender],axis=1)
print x.head()
'''
   CreditScore  Age  Tenure    Balance  ...  EstimatedSalary  Germany  Spain  Male
0          619   42       2       0.00  ...        101348.88        0      0     0
1          608   41       1   83807.86  ...        112542.58        0      1     0
2          502   42       8  159660.80  ...        113931.57        0      0     0
3          699   39       1       0.00  ...         93826.63        0      0     0
4          850   43       2  125510.82  ...         79084.10        0      1     0

[5 rows x 11 columns]'''

params={
 "learning_rate"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
 "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
 "min_child_weight" : [ 1, 3, 5, 7 ],
 "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
 "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
    
}

## Hyperparameter optimization using RandomizedSearchCV
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))
classifier=xgboost.XGBClassifier()

classifier=xgboost.XGBClassifier()

random_search=RandomizedSearchCV(classifier,param_distributions=params,n_iter=5,scoring='roc_auc',n_jobs=-1,cv=5,verbose=3)

from datetime import datetime
# Here we go
start_time = timer(None) # timing starts from this point for "start_time" variable
random_search.fit(x,y)
timer(start_time)

print random_search.best_estimator_
'''
XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.3, learning_rate=0.1,
       max_delta_step=0, max_depth=4, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)'''

print random_search.best_params_
'''
{'gamma': 0.2, 'learning_rate': 0.25, 'colsample_bytree': 0.4,
'max_depth': 4, 'min_child_weight': 1}'''

classifier=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bytree=0.5, gamma=0.3, learning_rate=0.1,
       max_delta_step=0, max_depth=4, min_child_weight=1, missing=None,
       n_estimators=100, n_jobs=1, nthread=None,
       objective='binary:logistic', random_state=0, reg_alpha=0,
       reg_lambda=1, scale_pos_weight=1, seed=None, silent=True,
       subsample=1)
    
from sklearn.model_selection import cross_val_score
score=cross_val_score(classifier,x,y,cv=10)

print score
'''
array([0.87012987, 0.86613387, 0.87012987, 0.867     , 0.862     ,
       0.852     , 0.864     , 0.87887888, 0.85885886, 0.85785786])'''

print score.mean()   # 0.8643

