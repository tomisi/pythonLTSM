#!/usr/bin/env python
# coding: utf-8

# In[5]:


# installing dependencies
import IPython
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings("ignore")
import re


# In[13]:


# importing dataset
data = pd.read_csv("C:/Users/Hp/Downloads/archive (1)/Corona_NLP_test.csv")


# In[ ]:


#checking the amount of rows and columns 
data.shape()


# In[89]:


#checking for null values
data.isnull().sum()


# In[90]:


# Filling in missing data as "remote" meaning users can be in any part of the world.
data.fillna("remote", inplace=True)


# In[85]:


#checking for null values
data.isnull().sum()


# In[86]:


data.drop_duplicates(inplace=True)


# In[87]:


data.to_csv("cleaned_data.csv", index=False)


# In[88]:


# statistical analysis
data.describe()


# In[113]:


# Visualization to understand sentiments distribution by loaction.

for location in data['Location'].unique():
    location_sentiment = data[data['Location'] == location]['Sentiment'].value_counts()
    location_sentiment = location_sentiment/location_sentiment.sum()
    location_sentiment.plot(kind='pie',autopct='%1.1f%%',title='Sentiment Distribution for {}'.format(location))
    plt.show()

