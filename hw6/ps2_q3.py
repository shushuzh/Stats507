# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Problem Set 2 - Question 3

# # Imports

# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd
from timeit import Timer
#from collections import defaultdict
from IPython.core.display import display, HTML
# 79: -------------------------------------------------------------------------

# ## (a)

# ### Step 1: Read the data and select variables (columns). 

# +
url1 = "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/DEMO_G.XPT"
url2 = "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/DEMO_H.XPT"
url3 = "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/DEMO_I.XPT"
url4 = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/DEMO_J.XPT"
url = [url1,url2,url3,url4]
cohort = ['G','H','I','J']

df = pd.DataFrame()
for i in range(4):
    df_adhoc = pd.read_sas(url[i])
    df_adhoc = df_adhoc[["SEQN", "RIDAGEYR", "RIDRETH3", "DMDEDUC2","DMDMARTL", 
                         "RIDSTATR", "SDMVPSU", "SDMVSTRA", "WTMEC2YR", 
                         "WTINT2YR",'RIAGENDR']]#Add gender
    df_adhoc['cohort'] = cohort[i]
    df = pd.concat([df,df_adhoc])
df
# -

# ### Step 2: rename columns

df = df.rename(columns = {"SEQN": "ids", 
                          "RIDAGEYR": "age", 
                          "RIDRETH3": "race/ethnicity", 
                          "DMDEDUC2": "education", 
                          "DMDMARTL": "marital_status",
                          'RIAGENDR': 'gender'}) #Add gender
df = df.rename(str.lower, axis='columns')
df

# ### Step 3: convert to proper types

df = df.convert_dtypes()
#convert 'age' to integer
df['age'] = df['age'].astype(int)
#convert 'ids', 'race/ethnicity', 'education', 'marital_status' into categorical data
for var in ['ids', 'race/ethnicity', 'education', 'marital_status']:
    df[var] = pd.Categorical(df[var])
print(df.dtypes)
print(df.head())

# ### Step 4: export data in pickle format

df.to_pickle('demographic_2011-2018.pkl')

# ## (b)

url1 = "https://wwwn.cdc.gov/Nchs/Nhanes/2011-2012/OHXDEN_G.XPT"
url2 = "https://wwwn.cdc.gov/Nchs/Nhanes/2013-2014/OHXDEN_H.XPT"
url3 = "https://wwwn.cdc.gov/Nchs/Nhanes/2015-2016/OHXDEN_I.XPT"
url4 = "https://wwwn.cdc.gov/Nchs/Nhanes/2017-2018/OHXDEN_J.XPT"
url = [url1,url2,url3,url4]
cohort = ['G','H','I','J']

df_oral = pd.DataFrame()
regex_stm = r"OHX.{2}TC|OHX.{2}CTC|SEQN|OHDDESTS"
for i in range(4):
    df_adhoc = pd.read_sas(url[i])
    new_df = df_adhoc.filter(regex=regex_stm, axis=1)
    new_df['cohort'] = cohort[i]
    df_oral = pd.concat([df_oral, new_df])
df_oral

# +
#df = pd.DataFrame()
#regex_stm = r"OHX.{2}TC|OHX.{2}CTC|SEQN|OHDDESTS"
#for test in test1, test2, test3, test4:
#    new_df = test.filter(regex=regex_stm, axis=1)
#    df = pd.concat([df, new_df])
#df
# -

# Rename columns
df_oral = df_oral.rename(columns = {"SEQN": "ids", 
                          "OHDDESTS": "dentition_status"}) 
#                          "OHX.{2}TC": "tooth_count", 
#                          "OHXxxCTC": "coronal_cavities"})
df_oral = df_oral.rename(str.lower, axis='columns')
df_oral

df = df.convert_dtypes()
#convert all variables into categorical data
for var in df.columns:
    df[var] = pd.Categorical(df[var])
df_oral

df_oral.to_pickle('oral_2011-2018.pkl')

# ## (c)
# - The number of cases in demographic dataset (i.e. (a)) is 39156.
# - The number of cases in dentition dataset (i.e. (b)) is 35909.

print(df.shape)
print(df_oral.shape)
