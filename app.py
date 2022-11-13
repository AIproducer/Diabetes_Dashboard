#import libraries
import streamlit as st
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix, recall_score, f1_score

# Load Data
df = pd.read_csv('D:\python_ ka _Chila\VS code\day 29\diabetes.csv')
# App loook
st.title('Diabetes Prediction App')
st.sidebar.header('Patient Data')
st.write(df.describe())


X = df.drop(['Outcome'], axis=1)
y =  df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# fuction
def user_report():
    Pregnancies = st.sidebar.slider('Pregnancies',0, 17,2)
    Glucose = st.sidebar.slider('Glucose',0,199,117)
    BloodPressure = st.sidebar.slider('BloodPressure',0,122,72)
    SkinThickness = st.sidebar.slider('SkinThickness',0,99,23)
    Insulin = st.sidebar.slider('Insulin',0,846,30)
    BMI = st.sidebar.slider('BMI',0,67,32)
    DiabetesPedigreeFunction = st.sidebar.slider('PDiabetesPedigreeFunction',0.243750,2.420000,0.372500)
    Age = st.sidebar.slider('Age',21.000000,81.000000,29.000000)

    #dictionary

    user_report_data = {
        'Pregnancies' : Pregnancies,
        'Glucose' : Glucose,
        'BloodPressure' : BloodPressure,
        'SkinThickness' : SkinThickness,
        'Insulin' : Insulin,
        'BMI' : BMI,
        'DiabetesPedigreeFunction' : DiabetesPedigreeFunction,
        'Age' : Age
        }

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data

user_data = user_report()
st.subheader('Pateint data')
st.write('user_data')



# Model
model = RandomForestClassifier()
model.fit(X_train, y_train)
user_result = model.predict(user_data)


# Visualize
st.title('Visualize Prediction')
#color 
if user_result[0] ==0:
    color = 'Green'
else:
    color = 'Red'

# Age vs Pregnancies
st.subheader('Age vs Pregnancies')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x=df['Age'], y=df['Pregnancies'], hue=df['Outcome'])
ax2 = sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 20, 2))
plt.title('0 = No Diabetes, 1 = Diabetes')
st.pyplot(fig_preg)

# Age vs Glucose line plot
st.subheader('Age vs Glucose')
fig_gluc = plt.figure()
ax1 = sns.lineplot(x=df['Age'], y=df['Glucose'], hue=df['Outcome'])
ax2 = sns.lineplot(x=user_data['Age'], y=user_data['Glucose'], color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 200, 20))
plt.title('0 = No Diabetes, 1 = Diabetes')
st.pyplot(fig_gluc)



# Glugose vs Pregnancies
st.subheader('Glucose vs Pregnancies')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x=df['Glucose'], y=df['Pregnancies'], hue=df['Outcome'])
ax2 = sns.scatterplot(x=user_data['Glucose'], y=user_data['Pregnancies'], color=color)
plt.xticks(np.arange(10, 200, 10))
plt.yticks(np.arange(0, 20, 2))
plt.title('0 = No Diabetes, 1 = Diabetes')
st.pyplot(fig_preg)


# age vs BMI line plot
st.subheader('Age vs BMI')
fig_bmi = plt.figure()
ax1 = sns.lineplot(x=df['Age'], y=df['BMI'], hue=df['Outcome'])
ax2 = sns.lineplot(x=user_data['Age'], y=user_data['BMI'], color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 70, 5))
plt.title('0 = No Diabetes, 1 = Diabetes')
st.pyplot(fig_bmi)



#outcome

if user_result[0] == 0:
    st.write('You are Healthy')
    st.balloons()
else:
    st.write('You have Diabetes')
    st.warning('Please consult a doctor')
st.title('Output')
st.sidebar.subheader('Accuracy Score')
st.write('Accoracy_Score:',accuracy_score(y_test, model.predict(X_test)))
st.write('Recall_Score:',recall_score(y_test, model.predict(X_test)))
st.write('F1_Score:',f1_score(y_test, model.predict(X_test)))
st.write('Confusion Matrix:',confusion_matrix(y_test, model.predict(X_test)))