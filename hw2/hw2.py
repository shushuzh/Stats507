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

# # Imports

# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd
from timeit import Timer
#from collections import defaultdict
from IPython.core.display import display, HTML
# 79: -------------------------------------------------------------------------

# # Question 0: Code Review

# ## (a) 
# For a list of tuples, the code finds (largest) tuples with distinct first elements. 

sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
op = []
for m in range(len(sample_list)):
    li = [sample_list[m]]
    for n in range(len(sample_list)):
        if (sample_list[m][0] == sample_list[n][0] and
                sample_list[m][-1] != sample_list[n][-1]):
            li.append(sample_list[n])
    op.append(sorted(li, key=lambda dd: dd[-1], reverse=True)[0])
res = list(set(op))
print(res)


# ## (b) Suggestions
# - The indent of the code should be carefully handled. To be more specific, indent four spaces for the code from "for n in range.." to "op.append(...".
# - Change the index to the last element instead of a specific one, i.e., change "sample_list\[m\]\[3\] != sample_list\[n\]\[3\]" to "sample_list\[m\]\[-1\] != sample_list\[n\]\[-1\]", and change "dd\[3\]" to "dd\[-1\]". 
# - Change variable names to be more literate. For example, change "li" to "pair_list"; change "op" to "represent". 
# - The code has O(n^2) time complexity by nested loop, whose computational costs are too high. It can be reduced by going over the entire list only once, i.e. with time complexity O(n). See 2(b) and 2(c) for detailed implementation.

# # Question 1

def list_tuple_generator(n, k=3, low=0, high=10):
    output = []
    for i in range(n):
        rng = np.random.default_rng(i)
        output.append(tuple(rng.integers(low=low, high=high, size=k)))
    return (output)


# Test
print(list_tuple_generator(3))
assert isinstance(list_tuple_generator(3), list)
assert isinstance(list_tuple_generator(3)[0], tuple)
assert isinstance(list_tuple_generator(3)[1], tuple)
assert isinstance(list_tuple_generator(3)[2], tuple)


# # Question 2

# ## (a)

def distinct_a(sample_list, key_ele=0, dist_ele=2):
    op = []
    for m in range(len(sample_list)):
        li = [sample_list[m]]
        for n in range(len(sample_list)):
            if (sample_list[m][key_ele] == sample_list[n][key_ele] and
                    sample_list[m][dist_ele] != sample_list[n][dist_ele]):
                li.append(sample_list[n])
        op.append(sorted(li, key=lambda dd: dd[dist_ele], reverse=True)[0])
    res = list(set(op))
    return(res)


sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8), (1, 9, 7)]
distinct_a(sample_list)


# ## (b)

def distinct_b(sample_list, key_ele=0, dist_ele=2):
    res = []
    for m in range(len(sample_list)):
        n = [n for n in range(len(res)) if sample_list[m][key_ele] == res[n][key_ele]]
        if len(n)==0:
            res.append(sample_list[m])
        else:
            res[n[0]] = max(sample_list[m], res[n[0]])
    return(sorted(res))


sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
distinct_b(sample_list)


# ## (c)
# I think my code in (b) suffices the requirement of this question. So I just copy and paste my code from (b).

def distinct_c(sample_list, key_ele=0):
    res = dict()
    for m in range(len(sample_list)):
        if sample_list[m][key_ele] in res:
            res[sample_list[m][key_ele]] = max(res[sample_list[m][key_ele]],
                                               sample_list[m])
        else:
            res[sample_list[m][key_ele]] = sample_list[m]
    res = sorted(list(res.values()))
    return(res)


sample_list = [(1, 3, 5), (0, 1, 2), (1, 9, 8)]
distinct_c(sample_list)

# ## (d)
# We can see from the camparison, both (b) and (c) works much better than the original method (a). Generally speaking, (c) performs better than (b). This is because (b) still have to loop over the result list in the process while (c) does not. 

size = range(5,100,5)
function = ['a','b','c']
table = pd.DataFrame(columns = function, index = size)
for n in size:
    for fun in function:
        n_mc = 1000 #draw n_mc samples
        time = [] #store computing time for n_mc samples
        for rep in range(n_mc):
            sample = list_tuple_generator(n)
            t = Timer("f(n)", globals={"f": eval("".join(["distinct_",fun])), "n": sample})
            time.append(t.timeit(1))
        table.at[n,fun] = round(np.mean(time) * 1e6, 1)
print(table)       

# # Question 3

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
                         "WTINT2YR"]]
    df_adhoc['cohort'] = cohort[i]
    df = pd.concat([df,df_adhoc])
df
# -

# ### Step 2: rename columns

df = df.rename(columns = {"SEQN": "ids", 
                          "RIDAGEYR": "age", 
                          "RIDRETH3": "race/ethnicity", 
                          "DMDEDUC2": "education", 
                          "DMDMARTL": "marital_status"})
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
