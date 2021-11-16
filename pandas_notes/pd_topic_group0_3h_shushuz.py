# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
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

# ## Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
# Add a bullet for each topic and link to the level 2 title header using 
# the exact title with spaces replaced by a dash. 
#
# + [Topic Title](#Topic-Title)
# + [Hierarchical Indexing](#Hierarchical~Indexing)

# ## Topic Title
# Include a title slide with a short title for your content.
# Write your name in *bold* on your title slide. 

# !/usr/bin/env python
# coding: utf-8

# # Question 2 in Problem Set 6
#
# *Stats 507, Fall 2021*
#
# Shihao Wu, PhD student in statistics
#
# wshihao@umich.edu

# ## Imports
#
# The remaining questions will use the following imports.

# In[3]:


import pandas as pd
import numpy as np


# ## Question 0 - Topics in Pandas
#
# For this question, please pick a topic - such as a function, class, method, recipe or idiom related to the pandas python library and create a short tutorial or overview of that topic. The only rules are below.
#
# 1. Pick a topic *not* covered in the class slides.
# 2. Do not knowingly pick the same topic as someone else.
# 3. Use bullet points and titles (level 2 headers) to create the equivalent of **3-5** “slides” of key points. They shouldn’t actually be slides, but please structure your key points in a manner similar to the class slides (viewed as a notebook).
# 4. Include executable example code in code cells to illustrate your topic.
#
# You do not need to clear your topic with me. If you want feedback on your topic choice, please discuss with me or a GSI in office hours.

# ## Topic: Missing Data in Pandas
#
# Shihao Wu, PhD student in statistics
#
# Reference: [https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)
#
# There are 4 "slides" for this topic.
#
#
# ## Missing data
# Missing data arises in various circumstances in statistical analysis. Consider the following example:

# In[4]:


# generate a data frame with float, string and bool values
df = pd.DataFrame(
    np.random.randn(5, 3),
    index=["a", "c", "e", "f", "h"],
    columns=["1", "2", "3"],
)
df['4'] = "bar"
df['5'] = df["1"] > 0

# reindex so that there will be missing values in the data frame
df2 = df.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])

df2


# The missing values come from unspecified rows of data.

# ## Detecting missing data
#
# To make detecting missing values easier (and across different array dtypes), pandas provides the <code>isna()</code> and <code>notna()</code> functions, which are also methods on Series and DataFrame objects:

# In[5]:


df2["1"]


# In[6]:


pd.isna(df2["1"])


# In[7]:


df2["4"].notna()


# In[8]:


df2.isna()


# ## Inserting missing data
#
# You can insert missing values by simply assigning to containers. The actual missing value used will be chosen based on the dtype.
#
# For example, numeric containers will always use <code>NaN</code> regardless of the missing value type chosen:

# In[9]:


s = pd.Series([1, 2, 3])
s.loc[0] = None
s


# Because <code>NaN</code> is a float, a column of integers with even one missing values is cast to floating-point dtype. pandas provides a nullable integer array, which can be used by explicitly requesting the dtype:

# In[10]:


pd.Series([1, 2, np.nan, 4], dtype=pd.Int64Dtype())


# Likewise, datetime containers will always use <code>NaT</code>.
#
# For object containers, pandas will use the value given:

# In[11]:


s = pd.Series(["a", "b", "c"])
s.loc[0] = None
s.loc[1] = np.nan
s


# ## Calculations with missing data
#
# Missing values propagate naturally through arithmetic operations between pandas objects.

# In[12]:


a = df2[['1','2']]
b = df2[['2','3']]
a + b


# Python deals with missing value for data structure in a smart way. For example:
#
# * When summing data, NA (missing) values will be treated as zero.
# * If the data are all <code>NA</code>, the result will be 0.
# * Cumulative methods like <code>cumsum()</code> and <code>cumprod()</code> ignore <code>NA</code> values by default, but preserve them in the resulting arrays. To override this behaviour and include <code>NA</code> values, use <code>skipna=False</code>.

# In[13]:


df2


# In[14]:


df2["1"].sum()


# In[15]:


df2.mean(1)


# In[16]:


df2[['1','2','3']].cumsum()


# In[17]:


df2[['1','2','3']].cumsum(skipna=False)


# Missing data is ubiquitous. Dealing with missing is unavoidable in data analysis. This concludes my topic here.


# -*- coding: utf-8 -*-
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

# ## Hierarchical Indexing
#
# - Shushu Zhang
# - shushuz@umich.edu
# - Reference is [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html) 
#
# Hierarchical / Multi-level indexing is very exciting as it opens the door to some quite sophisticated data analysis and manipulation, especially for working with higher dimensional data. In essence, it enables you to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures like Series (1d) and DataFrame (2d).

# ### Import

# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd
from timeit import Timer
from os.path import exists
from collections import defaultdict
from IPython.core.display import display, HTML
import math
from math import floor
import matplotlib.pyplot as plt 
import matplotlib as mpl
import matplotlib.cm as cm
from scipy.stats import norm, t, chi2_contingency, binom, bernoulli, beta, ttest_ind
from function import ci_mean, ci_prop
from warnings import warn
# 79: -------------------------------------------------------------------------

# ### Creating a MultiIndex (hierarchical index) object
# - The MultiIndex object is the hierarchical analogue of the standard Index object which typically stores the axis labels in pandas objects. You can think of MultiIndex as an array of tuples where each tuple is unique. 
# - A MultiIndex can be created from a list of arrays (using MultiIndex.from_arrays()), an array of tuples (using MultiIndex.from_tuples()), a crossed set of iterables (using MultiIndex.from_product()), or a DataFrame (using MultiIndex.from_frame()). 
# - The Index constructor will attempt to return a MultiIndex when it is passed a list of tuples.

# Constructing from an array of tuples
arrays = [
    ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
    ["one", "two", "one", "two", "one", "two", "one", "two"],
]
tuples = list(zip(*arrays))
tuples
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
index

# ### Manipulating the dataframe with MultiIndex
#
# - Basic indexing on axis with MultiIndex is illustrated as below.
# - The MultiIndex keeps all the defined levels of an index, even if they are not actually used.

# Use the MultiIndex object to construct a dataframe 
df = pd.DataFrame(np.random.randn(3, 8), index=["A", "B", "C"], columns=index)
print(df)
df['bar']

#These two indexing are the same
print(df['bar','one'])
print(df['bar']['one'])

print(df.columns.levels)  # original MultiIndex
print(df[["foo","qux"]].columns.levels)  # sliced

# ### Advanced indexing with hierarchical index
# - MultiIndex keys take the form of tuples. 
# - We can use also analogous methods, such as .T, .loc. 
# - “Partial” slicing also works quite nicely.

df = df.T
print(df)
print(df.loc[("bar", "two")])
print(df.loc[("bar", "two"), "A"])
print(df.loc["bar"])
print(df.loc["baz":"foo"])


# ### Using slicers
#
# - You can slice a MultiIndex by providing multiple indexers.
#
# - You can provide any of the selectors as if you are indexing by label, see Selection by Label, including slices, lists of labels, labels, and boolean indexers.
#
# - You can use slice(None) to select all the contents of that level. You do not need to specify all the deeper levels, they will be implied as slice(None).
#
# - As usual, both sides of the slicers are included as this is label indexing.

# +
def mklbl(prefix, n):
    return ["%s%s" % (prefix, i) for i in range(n)]


miindex = pd.MultiIndex.from_product(
    [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)]
)


micolumns = pd.MultiIndex.from_tuples(
    [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")], names=["lvl0", "lvl1"]
)


dfmi = (
    pd.DataFrame(
        np.arange(len(miindex) * len(micolumns)).reshape(
            (len(miindex), len(micolumns))
        ),
        index=miindex,
        columns=micolumns,
    )
    .sort_index()
    .sort_index(axis=1)
)

print(dfmi)
print(dfmi.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :])
