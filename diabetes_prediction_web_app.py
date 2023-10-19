# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:15:49 2023

@author: Madu
"""
import numpy as np
import pickle
import streamlit as strl
loaded_model = pickle.load(open('trained_model.sav', 'rb'))


#Creating a function for Prediction
def diabetes_prediction(input_data):

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'

def main():
    
    # giving a title to the app
    strl.title('Diabetes Prediction App')
    
    # getting input data from the user
    Pregnancies = strl.text_input('Number of pregnancies')
    Glucose = strl.text_input('Glucose Level')
    BloodPressure = strl.text_input('Blood Pressure Value')
    SkinThickness = strl.text_input('Skin Thickness Value')
    Insulin = strl.text_input('Insulin Level')
    BMI = strl.text_input('Body Mass Index Value')
    DiabetesPedigreeFunction = strl.text_input('Diabetes Pedigree Function Value')
    Age = strl.text_input('Age of the person')
    
    # code for prediction
    diagnosis = ''
    
    # creating a button for prediction
    
    if strl.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    strl.success(diagnosis)
    
    
    
    
# calling the main function to display the result    
if __name__ == '__main__':
    main()
        
        