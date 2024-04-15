#!/usr/bin/env python
# coding: utf-8

# ### Import relevant libraries

# In[6]:


import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin


# ### Create our own Scaler

# In[7]:


class CustomScaler(BaseEstimator,TransformerMixin): 
    
    def __init__(self,columns):
        self.scaler = StandardScaler()
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns])
        self.mean_ = np.array(np.mean(X[self.columns]))
        self.var_ = np.array(np.var(X[self.columns]))
        return self

    def transform(self, X, y=None, copy=None):
        init_col_order = X.columns
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]

        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# ### Build absenteeism_module class which has load and clean data function, predict functions

# In[8]:


class absenteeism_model():
    
    # Open model and scaler trained
    def __init__(self,model_file,scaler_file):
        with open('model','rb') as file_model , open('scaler','rb') as file_scaler:
            self.reg_model = pickle.load(file_model)
            self.reg_scaler = pickle.load(file_scaler)
            self.data = None
    
    # Load and clean data function
    def load_and_clean_data(self,data_file):
        
        # Load data
        df = pd.read_csv(data_file)
        
        # Create a checkpoint with raw_data
        self.df_for_predictions = df.copy()
        
        # drop unnecessary column
        df = df.drop(['ID'],axis=1)
        
        # create dummies 
        reason_for_absence = pd.get_dummies(df['Reason for Absence'],drop_first=True,dtype=int)
        reason_type_A = reason_for_absence.loc[:,0:14].max(axis=1)
        reason_type_B = reason_for_absence.loc[:,15:17].max(axis=1)
        reason_type_C = reason_for_absence.loc[:,18:21].max(axis=1)
        reason_type_D = reason_for_absence.loc[:,22:].max(axis=1)
        
        # Add 4 types of reason to data
        df = pd.concat([df,reason_type_A,reason_type_B,reason_type_C,reason_type_D],axis=1)
        
        # Drop Reason for Absence
        df = df.drop(['Reason for Absence'],axis=1)
        
        # Change column name
        column_names = ['Date', 'Transportation Expense','Distance to Work','Age','Daily Work Load Average','Body Mass Index', 'Education', 'Children', 'Pets','Type_1', 'Type_2', 'Type_3', 'Type_4']
        df.columns = column_names
        
        # Reorder column name
        column_names_reordered = ['Type_1', 'Type_2', 'Type_3', 'Type_4','Date', 'Transportation Expense','Distance to Work','Age','Daily Work Load Average','Body Mass Index', 'Education', 'Children', 'Pets']
        df = df[column_names_reordered]
        
        # Convert str date to timestamp
        df['Date'] = pd.to_datetime(df['Date'],format='%d/%m/%Y')
    
        # Extract month of the year
        include_month = []
        for i in range(df.shape[0]):
            include_month.append(df['Date'][i].month)
        df['Month'] = include_month
        
        # Extract weekday of Date column
        def extract_weekday(date):
            return date.weekday()                      
        df['Day of Week'] = df['Date'].apply(extract_weekday)
        df = df.drop(['Date'],axis=1)
                                    
        # Change order of columns 
        column_names_reordered_1 = ['Type_1', 'Type_2', 'Type_3', 'Type_4','Month', 'Day of Week', 'Transportation Expense',
                                   'Distance to Work', 'Age', 'Daily Work Load Average',
                                   'Body Mass Index', 'Education', 'Children', 'Pets']
        df = df[column_names_reordered_1]
                                 
        # Get dummies Education
        df['Education'] = df['Education'].map({1:0,3:1,2:1,4:1})
        
        # Process NaN data (if have)
        df = df.fillna(value=0)
        
        # Save if user need to call preprocessed data                   
        self.preprocessed_data = df.copy()
        
        # Scaled data
        self.data = self.reg_scaler.transform(df)
    
    # Predict probability of the result
    def predicted_probability(self):
        if (self.data is not None):
            return self.reg_model.predict_proba(self.data)
    
    # Predict 1 or 0 category
    def predicted_category_output(self):
        if (self.data is not None):
            return self.reg_model.predict(self.data)
    
    # Predict all
    def predict_outputs_with_inputs(self):
        if (self.data is not None):
            self.preprocessed_data['Probability'] = self.reg_model.predict_proba(self.data)[:,1]
            self.preprocessed_data['Prediction'] = self.reg_model.predict(self.data)
            return self.preprocessed_data


# In[ ]:




