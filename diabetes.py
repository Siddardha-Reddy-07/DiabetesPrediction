#importing essential libraries
import pandas as pd
import numpy as np
import pickle

#Loading the dataset
df=pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFuncion as DPF
df=df.rename(columns={'DiabetesPedigreeFunction':'DPF'})

# Replacing the 0 values from ['GLucose','BloodPressure','SkinThickness','Insulin','BMI']
df_copy=df.copy(deep=True)
df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']]=df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.nan)

# Replacing NaN value by mean , median depending upon distribution
df_copy['Glucose'].fillna(df['Glucose'].mean(),inplace=True)
df_copy['BloodPressure'].fillna(df['BloodPressure'].mean(),inplace=True)
df_copy['SkinThickness'].fillna(df['SkinThickness'].median(),inplace=True)
df_copy['Insulin'].fillna(df['Insulin'].median(),inplace=True)
df_copy['BMI'].fillna(df['BMI'].median(),inplace=True)

# Model Building
from sklearn.model_selection import train_test_split
x=df.drop(columns='Outcome')
y=df['Outcome']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=0)

# Creating Random Forest model
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier(n_estimators=20)
classifier.fit(x_train,y_train)

# creating pickle file for the classifier
filename='diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier,open(filename,'wb'))

