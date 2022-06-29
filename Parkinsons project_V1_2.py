#!/usr/bin/env python
# coding: utf-8

# # 1.Data Preprocessing

# <div class="alert alert-block alert-info">
# <b>Load dataset and import librairies
# </div>

# ### Import NumPy for numerical calculation, Pandas for handling data and visualization with Seaborn and Matplotlib

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams
rcParams['figure.figsize']=(12,6)
sns.set
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ### Import the dataset

# In[91]:


parkinsons_data=pd.read_csv('parkinsons.csv')


# In[92]:


# Using shape function, we can observe the dimensions of the data
parkinsons_data.shape


# <font color='green'>There are 24 columns and 195 observations</font>

# In[93]:


# We can observe the dataset using the head()function, which returns the first five records from the dataset
parkinsons_data.head()


# <font color='green'>This project is a classification problem, from which we predict the binary variable "status" which can either be ill with Parkinson disease or not </font>

# In[94]:


# The info() method shows some of the characteristics of the data 
parkinsons_data.info()


# <font color='green'>We can see that we have mainly float or numeric data except for "name" and "status" columns</font>

# <div class="alert alert-block alert-info">
# <b>Statistical insights
# </div>

# In[95]:


parkinsons_data.describe()


# <div class="alert alert-block alert-info">
# <b>Exploratory Data Analysis (EDA)
# </div>

# #### Let's build subplots of the features with boxplot or whisker plot to see the minimum, 1srt quantile, median and max.
# #### Each time the feature is related to the status (0 or 1 for healthy or Parkinson disease)

# In[96]:


fig, axes=plt.subplots(5,5,figsize=(15,15))
axes=axes.flatten()

for i in range(1,len(parkinsons_data.columns)-1):
    sns.boxplot(x='status', y=parkinsons_data.iloc[:,i], data=parkinsons_data, orient='v', ax=axes[i])
plt.tight_layout()
plt.show()


# <font color='green'>Thanks to these subplots we can easily see the outliers, the points that lie outside the whiskers.</font>

# In[97]:


# Let's have a closer look to the "PPE" variable
parkinsons_data.boxplot(column=['PPE'])
plt.show
# There are some outliers between 0.4 and 0.5 and beyond 0.5


# <div class="alert alert-block alert-info">
# <b></b> Data Cleaning

# ### Removing duplicates and finding missing values are important, otherwise our models can lead us to incorrect conclusions

# In[98]:


duplicate_Values=parkinsons_data.duplicated()
print(duplicate_Values.sum())
parkinsons_data[duplicate_Values]


# <font color='green'>There are no duplicate variables</font>

# In[99]:


print(parkinsons_data.isnull().sum())


# <font color='green'>Luckily, this dataset does not contain any missing values</font>

# <div class="alert alert-block alert-info">
# <b>Correlation Analysis
# </div>

# In[100]:


corr= parkinsons_data.corr()
corr


# <font color='green'>We don't have a clear visualization</font>

# ### We create a heatmap using Seaborn

# In[101]:


sns.heatmap(parkinsons_data.corr(),annot=True,cmap='RdYlGn')


# <font color='green'>As it shown, it is not hard to find following pairs of highly correlated features:<br> Spread1 and PPE = 0,96</font>

# ### For a better visualization let's see below

# In[129]:


sns.heatmap(corr[(corr>0.9)],annot=True,cmap='PuBu')


# <font color='green'>We can also check the correlation between "spread1" and "PPE" variables thanks to "regplot" which plots the scartterplot plus the fitted regression line for the data.</font>

# In[130]:


sns.regplot(x='spread1', y='PPE', data=parkinsons_data)


# ### Let's go deeper with the P-value to know the significance of the correlation estimate when p-value is <br> < 0.001 leads to strong evidence  <br> < 0.05 leads to moderate evidence <br> < 0.1 leads to weak evidence <br> > 0.1 leads to no evidence

# In[131]:


from scipy import stats


# In[132]:


pearson_coef, p_value = stats.pearsonr(parkinsons_data['spread1'], parkinsons_data['PPE'])
print ("The Pearson Correlation coefficient is", pearson_coef, " with a P-value of P =", p_value)


# <font color='green'>The linear relationship is strong between "spread1" and "PPE"(app. 0.96) and the correlation is statistically very significant</font>

# # 2.Modeling

# <div class="alert alert-block alert-info">
# <b>Let's explore some models, tune it to optimize its performance with the better predictability rate
# </div>

# In[133]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import tree
from os import system
from sklearn import metrics


# In[134]:


# As we have seen previously, we can remove the non-numeric columns "name" and "status"
X = parkinsons_data.drop(['status', 'name'], axis = 1)
Y = parkinsons_data.status


# ### Decision Tree Model

# In[135]:


# Splitting Data into 70% Training data and 30% Testing Data
X_train, X_test, y_train,  y_test = train_test_split(X, Y,train_size=0.7, test_size=0.3, random_state=42)
print(len(X_train)),print(len(X_test))


# In[136]:


# Applying decision tree model
decision_tree = DecisionTreeClassifier(criterion='entropy',max_depth=6,random_state=100,min_samples_leaf=5)
decision_tree.fit(X_train, y_train)
decision_tree.score(X_test , y_test) 


# In[137]:


y_pred = decision_tree.predict(X_test)
confusion_matrix(y_test,y_pred)


# In[138]:


count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples in Decision Tree: {}'.format(count_misclassified))


# <font color='green'>We've got 9 outliers samples</font>

# ### Random Forest

# In[139]:


randomforest = RandomForestClassifier(n_estimators = 50)
randomforest = randomforest.fit(X_train, y_train)
y_pred = randomforest.predict(X_test)
randomforest.score(X_test , y_test)


# In[140]:


count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples in Random Forest: {}'.format(count_misclassified))


# <font color='green'>We've got 4 outliers samples</font>

# In[141]:


feature_imp = pd.Series(randomforest.feature_importances_,index=X.columns).sort_values(ascending=False)
feature_imp

# Creating a bar plot
sns.barplot(x=feature_imp, y=feature_imp.index)

# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.legend()
plt.show()


# ### KNN

# In[142]:


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
knn.score(X_test,y_test)


# In[143]:


y_pred = knn.predict(X_test)
count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples in KNN: {}'.format(count_misclassified))


# <font color='green'>We've got 6 outliers samples</font>

# ### ADABoost

# In[144]:


adb = AdaBoostClassifier( n_estimators= 50)
adb = adb.fit(X_train,y_train)
y_pred = adb.predict(X_test)
adb.score(X_test , y_test)


# In[145]:


count_misclassified = (y_test != y_pred).sum()
print('Misclassified samples in Ada Boosting: {}'.format(count_misclassified))


# <font color='green'>We've got 9 outliers samples</font>

# In[146]:


from sklearn.dummy import DummyClassifier
from sklearn.metrics import plot_confusion_matrix
clf_dummy= DummyClassifier(random_state=42)
clf_dummy.fit(X_train, y_train)
y_pred = clf_dummy.predict(X_test)
plot_confusion_matrix(estimator=clf_dummy, X=X_test, y_true=y_test, normalize='true', cmap='Blues')


# In[147]:


y_train.value_counts(normalize=True)


# <font color='green'>We have more people with Parkinson disease than people with not Parkinson <br> So the dummy classifier is predicting more people with the disease</font>

# # Performance evaluation

# <div class="alert alert-block alert-info">
# <b>With the confusion matrix we can have some metrics such as the accuracy, F1 score and the precision
# </div>

# In[148]:


print(accuracy_score(y_test, y_pred))


# In[149]:


cm = confusion_matrix(y_test, y_pred)
print (cm)


# In[150]:


print(classification_report(y_test, y_pred))

