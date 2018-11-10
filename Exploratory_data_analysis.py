# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 11:07:43 2018

@author: Immanuel
"""

#importig the requisite python libraries
import pandas as pd
import numpy as np                     # For mathematical calculations
import seaborn as sns                  # For data visualization
import matplotlib.pyplot as plt        # For plotting graphs
%matplotlib inline
import warnings                        # To ignore any warnings
warnings.filterwarnings("ignore")
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"

#Loading the dataset
df = pd.read_csv("C:/users/Immanuel/downloads/black_friday.csv")

#Checking the dimensions of the dataset
df.shape

#Firstly, let us look at the features present in our dataset and then
df.columns

#Lets look at their datatypes
#Based on the output of the code below, you should be able to formulate descriptions of the various features
df.dtypes

#Lets analyse the measures of central tendencies of the data

df.describe()


#Univariate Analysis
#Exploring the Target Variable. Our target variable is the Purchase column
df['Purchase'].value_counts()

# Normalize can be set to True to print proportions instead of number 
df['Purchase'].value_counts(normalize=True)

#Lets plot a simple bar graph on the target variable
df['Purchase'].value_counts().hist()


#Lets plot a simple bar graph on the target variable
df['Purchase'].value_counts().plot.bar();


#Now let us visualize the independent variables
plt.figure(1)
plt.subplot(221)
df['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10), title= 'Gender')

plt.subplot(222)
df['Age'].value_counts(normalize=True).plot.bar(title= 'Age')

plt.subplot(223)
df['Occupation'].value_counts(normalize=True).plot.bar(title= 'Occupation')

plt.subplot(224)
df['City_Category'].value_counts(normalize=True).plot.bar(title= 'City_Category')

plt.show();

plt.figure(1)
plt.subplot(121)
sns.distplot(df['Product_Category_1']);

plt.subplot(122)
df['Purchase'].plot.box(figsize=(16,5))

plt.show();



#Analysing more than one variable
#We can take a look at Purchases by Product category

df.boxplot(column='Purchase', by = 'Product_Category_1')
plt.suptitle("");



#Analysing more than one variable
#We can take a look at Purchases by Product category

df.boxplot(column='Purchase', by = 'Gender')
plt.suptitle("");


#Below we look at the relationship of more than one Independent variable

Gender=pd.crosstab(df['Gender'],df['Product_Category_1'])
Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(10,10))


#Analysing more than one variable
#We can take a look at Purchases by Product category
df.boxplot(column='Purchase', by = 'Marital_Status')
plt.suptitle("")


#Creation of purchase categories and analysing them per Marital Status
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
df['Purchase']=pd.cut(df['Purchase'],bins,labels=group)

Purchase=pd.crosstab(df['Purchase'],df['Marital_Status'])
Purchase.div(Purchase.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Purchase')
P = plt.ylabel('Percentage')


#Creation of purchase categories and analysing them per Gender
bins=[0,2500,4000,6000,81000]
group=['Low','Average','High', 'Very high']
df['Purchase']=pd.cut(df['Purchase'],bins,labels=group)

Purchase=pd.crosstab(df['Purchase'],df['Gender'])
Purchase.div(Purchase.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
plt.xlabel('Purchase')
P = plt.ylabel('Proportion on a scale of 0 to 1')


#Checking the distribution of Purchase column data
df['Purchase'].hist(bins=20)


#Normalization of data
df['Purchase'] = np.log(df['Purchase'])
df['Purchase'].hist(bins=20)


#Checking for Missing values in the dataset
df.isnull().sum()


#Filling the two product category columns with the mode

df['Product_Category_2'].fillna(df['Product_Category_2'].mode()[0], inplace=True)
df['Product_Category_3'].fillna(df['Product_Category_3'].mode()[0], inplace=True)


#Checking to confirm that Missing values in the dataset have been replaced
df.isnull().sum()


df.plot(x='Occupation', y='Purchase', style='*')  
plt.title('Purchase based on Age')  
plt.xlabel('Age')  
plt.ylabel('Purchase')  
plt.show()  


#import the library LabelEncoder
from sklearn.preprocessing import LabelEncoder
#Create a list with categorical predictors
cat_var =['Gender','Age','City_Category']
#Initiate LabelEncoder
le = LabelEncoder() 
#A for loop to transform the categorical values to numerical values
for n in cat_var:
    df[n] = le.fit_transform(df[n])
	
	
df.dtypes



#Time to build the model (Linear regression)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('E:/Training/AI Saturdays/Ambassador/Strathmore/Lecture Notes/Week 4/black_friday.csv')
print(data.shape)
data.head()

#drop all rows with missing values
data.dropna(inplace=True)


#import the library LabelEncoder
from sklearn.preprocessing import LabelEncoder
#Create a list with categorical predictors
cat_var =['Gender','Age','City_Category']
#cat_var =['Gender','Married','Education','Self_Employed','Property_Area','Loan_Status']
#Initiate LabelEncoder
le = LabelEncoder() 
#A for loop to transform the categorical values to numerical values
for n in cat_var:
    data[n] = le.fit_transform(data[n])

#Checking for the type of the predictors afterwards
data.dtypes

#Getting the variables to an array.
Gender = data['Gender'].values
Age = data['Age'].values
Purchase = data['Purchase'].values	


# Ploting the scores as scatter plot
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(Gender, Age, Purchase, color='#ef1234')
plt.show()

#Now we generate our parameters(the theta values)
m = len(Gender)
x0 = np.ones(m)
X = np.array([x0, Gender, Age]).T
# Initial Coefficients
B = np.array([0, 0, 0])
Y = np.array(Purchase)
alpha = 0.0001

#We’ll define our cost function.
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J

inital_cost = cost_function(X, Y, B)
print("Initial Cost")
print(inital_cost)

#Defining the Gradient Descent
def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history

# 100 Iterations
newB, cost_history = gradient_descent(X, Y, B, alpha, 100)

# New Values of B
print("New Coefficients")
print(newB)

# Final Cost of new B
print("Final Cost")
print(cost_history[-1])

# Model Evaluation - RMSE
def rmse(Y, Y_pred):
    rmse = np.sqrt(sum((Y - Y_pred) ** 2) / len(Y))
    return rmse

# Model Evaluation - R2 Score
def r2_score(Y, Y_pred):
    mean_y = np.mean(Y)
    ss_tot = sum((Y - mean_y) ** 2)
    ss_res = sum((Y - Y_pred) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2

Y_pred = X.dot(newB)

print("RMSE")
print(rmse(Y, Y_pred))
print("R2 Score")
print(r2_score(Y, Y_pred))


