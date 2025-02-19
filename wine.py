
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
wine = pd.read_csv('winequality-red.csv')
wine

wine.head()

wine.tail()

wine.shape # Total no of rows and columns

wine.info()

wine.isnull()

wine.isnull().sum() # no of null values

### Data Analysis and Visualization

# statistical measure of the dataset
wine.describe()

#number of values for each quality
sns.countplot(x='quality', data= wine)

#volatile acidity vs quality ---> volatile acidity is indirectly propostional to quality
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='volatile acidity',data=wine)

# citric acid vs quality  ---> citric acid is directly propotional to quality
plot=plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='citric acid',data=wine)

# residual sugar vs quality
plot = plt.figure(figsize=(5,5))
sns.barplot(x='quality',y='residual sugar',data=wine)

correlation = wine.corr()

# construct a heatmap to find the correlation between columns
plt.figure(figsize=(10,10))
sns.heatmap(correlation,cbar=True,square=True,annot=True,annot_kws={'size':8},cmap='Blues')

#Data Preprocessing

# seperate the data and class label
X = wine.drop('quality',axis=1)
print(X)

#Label Binarization
y= wine['quality'].apply(lambda y_value:1 if y_value >= 7 else 0) #lambda--> replace the values
print(y)

## Train & Test split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=3)

#Model Training: Random Forest Classifier ( It is an ensemble of decision tree)

model = RandomForestClassifier()
model.fit(X_train,y_train)

#Model Evualtion

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,y_test)

print('Accuracy:',test_data_accuracy)
#Building a Predictive system
st.title('Wine Quality Prediction')
input_text = st.text_input('Enter the values:')
input_text_list = input_text.split(',')
if st.button('Predict'):
    input_data_as_numpy_array = np.asarray(input_text_list, dtype=float)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = model.predict(input_data_reshaped)
    if prediction[0] == 1:
        st.write('The wine quality is Good')
    else:
        st.write('The wine quality is Bad')
