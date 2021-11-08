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

# # Import

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

# # Question 0: Hierarchical indexing (MultiIndex)
# (Reference is [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html)). 
#
# Hierarchical / Multi-level indexing is very exciting as it opens the door to some quite sophisticated data analysis and manipulation, especially for working with higher dimensional data. In essence, it enables you to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures like Series (1d) and DataFrame (2d).

# ## Creating a MultiIndex (hierarchical index) object
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

# ## Manipulating the dataframe with MultiIndex
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

# ## Advanced indexing with hierarchical index
# - MultiIndex keys take the form of tuples. 
# - We can use also analogous methods, such as .T, .loc. 
# - “Partial” slicing also works quite nicely.

df = df.T
print(df)
print(df.loc[("bar", "two")])
print(df.loc[("bar", "two"), "A"])
print(df.loc["bar"])
print(df.loc["baz":"foo"])


# ## Using slicers
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
# -

# # Question 1
#
# ## (a)

# Import data file: ------------------------------------------------------
path = './'
demo_file = path + '/demo.feather'
ohx_file = path + '/ohx.feather'
demo = pd.read_feather(demo_file)
ohx = pd.read_feather(ohx_file)

# ## (b) 
# id (from SEQN)
# gender
# age
# under_20 if age < 20
# college - with two levels:
# ‘some college/college graduate’ or
# ‘No college/<20’ where the latter category includes everyone under 20 years of age.
# exam_status (RIDSTATR)
# ohx_status - (OHDDESTS)

# Choose variables needed
demo_less = demo[["id","gender","age","education","exam_status"]]
ohx_less = ohx[["id","dentition_status"]]
# Merge two datasets
data = pd.merge(demo_less,ohx_less,on='id',how='left')
# Create "under_20" variable
data["under_20"] = data["age"]<20
# Create "college" variable
data['college'] = np.where(
    (data['education']=='Some college or AA degree')\
     |(data['education']=='College graduate or above')\
     &(data["age"] >=20),
    'some college/college graduate','No college/<20')
data = data.drop(columns="education")
# Create "ohx" variable
data['ohx'] = np.where(
    (data['exam_status']=='Both interviewed and MEC examined')\
    &(data['dentition_status']=='Complete'),
    'complete','missing'
)
# Rename
data = data.rename(columns = {'dentition_status':'ohx_status'})
data

# ## (c)
# - Number of subjects removed: 1757;
# - Number of subjects left: 37399. 

data = data[(data['exam_status']=="Both interviewed and MEC examined")]

# ## (d)

# +
table = pd.DataFrame()
# For categorical varibles
for var in ['gender','under_20','college']:
    # Calculate the size for each of the four groups
    count = data.groupby([var, 'ohx']).size()
    percent = count/data.groupby(var).size()
    df = pd.concat([count,percent],axis=1)
    tab = df.apply(lambda x: 
               '{0:4.0f}({1:.2%})'.format(
                        x[0],
                        x[1]),axis=1)
    # Convert the 1*4 table to 2*2
    tab = tab.reset_index()
    tab = tab.pivot_table(0, var, 'ohx', aggfunc=lambda x: " ".join(x))
    # Convert the 1*4 table to 2*2 for count
    count = pd.DataFrame(count).reset_index()
    count = count.pivot_table(0, var, 'ohx')
    # p value
    _, p, _, _ = chi2_contingency(count)
    tab['p value'] = p
    table = pd.concat([table,tab])
    
# Add continuous variable (i.e., 'age')
age = data.groupby(['ohx']).agg({'age':["mean","std"]})
age = round(age.T,2)
## p value
data_age = data['age']
_, age['p value'] = ttest_ind(data_age[data['ohx']=="complete"],data_age[data['ohx']=="missing"])
table = pd.concat([table,age])


index = [
    np.array(['gender','gender','under_20','under_20','college','college','age','age']),
    np.array(["Female", "Male", "False", "True", "No collge/<20", "some college/college graduate",'mean','std'])
]
table.index = index
table = table.style.format(formatter={'p value': "{:.2e}"})
table.columns.name = 'group'
table
# -

# add a caption: --------------------------------------------------------------
cap = """
<caption style="text-align:justify; caption-side:bottom"> <b> Table 1.</b> 
<em> Summary of the size of different variables with respect to dentition status
</caption>
"""
t1 = table.to_html()
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'
display(HTML(tab1))

# # Question 2
#
# ## (a)

#Create large dataframe for estimated confidence level
alpha_range = [0.8,0.9,0.95]
method_range = ["Normal","CP","Jeffrey","AC"]
p_range = np.arange(0.05,0.5,0.05)
n_range = range(200,2000,200)
iterables = [alpha_range, method_range,n_range]
index = pd.MultiIndex.from_product(iterables, names=["alpha","method","n"])
conf_level = pd.DataFrame(np.random.randn(len(index), len(p_range)), index=index, columns=p_range)
#Create large dataframe for width of confidence interval
width = pd.DataFrame(np.random.randn(len(index), len(p_range)), index=index, columns=p_range)

for alpha in alpha_range:
    for method in method_range:
        for p in p_range:
            z = norm.ppf(1-(1-alpha)/2)
            n_min = (1/0.005*z*np.sqrt(p*(1-p)))**2 
            for n in n_range:
                level = 0
                wid = 0
                for sim in range(int(n_min)):  
                    x = bernoulli.rvs(p,size = n)
                    res = ci_prop(
                        x,
                        alpha,
                        str_fmt=None,
                        #"{mean:.2f} [{level:.0f}%: ({lwr:.2f}, {upr:.2f})]",
                        method=method
                    )
                    level = level + (res['lwr']<p<res['upr'])
                    wid = wid + res['upr']- res['lwr']
                conf_level.loc[(alpha, method, n),p] = level/n_min
                width.loc[(alpha, method, n),p] = wid/n_min

X, Y = np.meshgrid(p_range,n_range)

# ### Contour Plot for Estimated Confidence Level

# +
#alpha is 0.8
alpha = 0.8
fig0 = plt.figure()
fig0.suptitle('Estimated Confidence Level with Alpha=0.8')
ax0 = fig0.add_subplot(2, 2, 1)
ax1 = fig0.add_subplot(2, 2, 2)
ax2 = fig0.add_subplot(2, 2, 3) 
ax3 = fig0.add_subplot(2, 2, 4) 
## Normal
CS0 = ax0.contour(X,Y,conf_level.loc[(alpha, "Normal")])
ax0.clabel(CS0, fontsize=10)
ax0.set_title('Normal Approximation')
ax0.set_xlabel('p')
ax0.set_ylabel('n')
## CP
CS1 = ax1.contour(X,Y,conf_level.loc[(alpha, "CP")])
ax1.clabel(CS1, fontsize=10)
ax1.set_title('Clopper-Pearson interval')
ax1.set_xlabel('p')
ax1.set_ylabel('n')
## Jeffrey
CS2 = ax2.contour(X,Y,conf_level.loc[(alpha, "Jeffrey")])
ax2.clabel(CS2, fontsize=10)
ax2.set_title('Jeffrey\'s method')
ax2.set_xlabel('p')
ax2.set_ylabel('n')
## AC
CS3 = ax3.contour(X,Y,conf_level.loc[(alpha, "AC")])
ax3.clabel(CS3, fontsize=10)
ax3.set_title('Agresti-Coull estimates')
ax3.set_xlabel('p')
ax3.set_ylabel('n')

plt.tight_layout()
display()

# +
#alpha is 0.9
alpha = 0.9
fig0 = plt.figure()
fig0.suptitle('Estimated Confidence Level with Alpha=0.9')
ax0 = fig0.add_subplot(2, 2, 1)
ax1 = fig0.add_subplot(2, 2, 2)
ax2 = fig0.add_subplot(2, 2, 3) 
ax3 = fig0.add_subplot(2, 2, 4) 
## Normal
CS0 = ax0.contour(X,Y,conf_level.loc[(alpha, "Normal")])
ax0.clabel(CS0, fontsize=10)
ax0.set_title('Normal Approximation')
ax0.set_xlabel('p')
ax0.set_ylabel('n')
## CP
CS1 = ax1.contour(X,Y,conf_level.loc[(alpha, "CP")])
ax1.clabel(CS1, fontsize=10)
ax1.set_title('Clopper-Pearson interval')
ax1.set_xlabel('p')
ax1.set_ylabel('n')
## Jeffrey
CS2 = ax2.contour(X,Y,conf_level.loc[(alpha, "Jeffrey")])
ax2.clabel(CS2, fontsize=10)
ax2.set_title('Jeffrey\'s method')
ax2.set_xlabel('p')
ax2.set_ylabel('n')
## AC
CS3 = ax3.contour(X,Y,conf_level.loc[(alpha, "AC")])
ax3.clabel(CS3, fontsize=10)
ax3.set_title('Agresti-Coull estimates')
ax3.set_xlabel('p')
ax3.set_ylabel('n')

plt.tight_layout()
display()

# +
#alpha is 0.95
alpha = 0.95
fig0 = plt.figure()
fig0.suptitle('Estimated Confidence Width with Alpha=0.95')
ax0 = fig0.add_subplot(2, 2, 1)
ax1 = fig0.add_subplot(2, 2, 2)
ax2 = fig0.add_subplot(2, 2, 3) 
ax3 = fig0.add_subplot(2, 2, 4) 
## Normal
CS0 = ax0.contour(X,Y,conf_level.loc[(alpha, "Normal")])
ax0.clabel(CS0, fontsize=10)
ax0.set_title('Normal Approximation')
ax0.set_xlabel('p')
ax0.set_ylabel('n')
## CP
CS1 = ax1.contour(X,Y,conf_level.loc[(alpha, "CP")])
ax1.clabel(CS1, fontsize=10)
ax1.set_title('Clopper-Pearson interval')
ax1.set_xlabel('p')
ax1.set_ylabel('n')
## Jeffrey
CS2 = ax2.contour(X,Y,conf_level.loc[(alpha, "Jeffrey")])
ax2.clabel(CS2, fontsize=10)
ax2.set_title('Jeffrey\'s method')
ax2.set_xlabel('p')
ax2.set_ylabel('n')
## AC
CS3 = ax3.contour(X,Y,conf_level.loc[(alpha, "AC")])
ax3.clabel(CS3, fontsize=10)
ax3.set_title('Agresti-Coull estimates')
ax3.set_xlabel('p')
ax3.set_ylabel('n')

plt.tight_layout()
display()
# -

# ## (b)
# ### Contour plot for Estimated Width of Confidence Interval

# +
#alpha is 0.8
alpha = 0.8
fig0 = plt.figure()
fig0.suptitle('Estimated Confidence Width with Alpha=0.8')
ax0 = fig0.add_subplot(2, 2, 1)
ax1 = fig0.add_subplot(2, 2, 2)
ax2 = fig0.add_subplot(2, 2, 3) 
ax3 = fig0.add_subplot(2, 2, 4) 
## Normal
CS0 = ax0.contour(X,Y,width.loc[(alpha, "Normal")])
ax0.clabel(CS0, fontsize=10)
ax0.set_title('Normal Approximation')
ax0.set_xlabel('p')
ax0.set_ylabel('n')
## CP
CS1 = ax1.contour(X,Y,width.loc[(alpha, "CP")])
ax1.clabel(CS1, fontsize=10)
ax1.set_title('Clopper-Pearson interval')
ax1.set_xlabel('p')
ax1.set_ylabel('n')
## Jeffrey
CS2 = ax2.contour(X,Y,width.loc[(alpha, "Jeffrey")])
ax2.clabel(CS2, fontsize=10)
ax2.set_title('Jeffrey\'s method')
ax2.set_xlabel('p')
ax2.set_ylabel('n')
## AC
CS3 = ax3.contour(X,Y,width.loc[(alpha, "AC")])
ax3.clabel(CS3, fontsize=10)
ax3.set_title('Agresti-Coull estimates')
ax3.set_xlabel('p')
ax3.set_ylabel('n')

plt.tight_layout()
display()

# +
#alpha is 0.9
alpha = 0.9
fig0 = plt.figure()
fig0.suptitle('Estimated Confidence Width with Alpha=0.9')
ax0 = fig0.add_subplot(2, 2, 1)
ax1 = fig0.add_subplot(2, 2, 2)
ax2 = fig0.add_subplot(2, 2, 3) 
ax3 = fig0.add_subplot(2, 2, 4) 
## Normal
CS0 = ax0.contour(X,Y,width.loc[(alpha, "Normal")])
ax0.clabel(CS0, fontsize=10)
ax0.set_title('Normal Approximation')
ax0.set_xlabel('p')
ax0.set_ylabel('n')
## CP
CS1 = ax1.contour(X,Y,width.loc[(alpha, "CP")])
ax1.clabel(CS1, fontsize=10)
ax1.set_title('Clopper-Pearson interval')
ax1.set_xlabel('p')
ax1.set_ylabel('n')
## Jeffrey
CS2 = ax2.contour(X,Y,width.loc[(alpha, "Jeffrey")])
ax2.clabel(CS2, fontsize=10)
ax2.set_title('Jeffrey\'s method')
ax2.set_xlabel('p')
ax2.set_ylabel('n')
## AC
CS3 = ax3.contour(X,Y,width.loc[(alpha, "AC")])
ax3.clabel(CS3, fontsize=10)
ax3.set_title('Agresti-Coull estimates')
ax3.set_xlabel('p')
ax3.set_ylabel('n')

plt.tight_layout()
display()

# +
#alpha is 0.95
alpha = 0.95
fig0 = plt.figure()
fig0.suptitle('Estimated Confidence Width with Alpha=0.95')
ax0 = fig0.add_subplot(2, 2, 1)
ax1 = fig0.add_subplot(2, 2, 2)
ax2 = fig0.add_subplot(2, 2, 3) 
ax3 = fig0.add_subplot(2, 2, 4) 
## Normal
CS0 = ax0.contour(X,Y,width.loc[(alpha, "Normal")])
ax0.clabel(CS0, fontsize=10)
ax0.set_title('Normal Approximation')
ax0.set_xlabel('p')
ax0.set_ylabel('n')
## CP
CS1 = ax1.contour(X,Y,width.loc[(alpha, "CP")])
ax1.clabel(CS1, fontsize=10)
ax1.set_title('Clopper-Pearson interval')
ax1.set_xlabel('p')
ax1.set_ylabel('n')
## Jeffrey
CS2 = ax2.contour(X,Y,width.loc[(alpha, "Jeffrey")])
ax2.clabel(CS2, fontsize=10)
ax2.set_title('Jeffrey\'s method')
ax2.set_xlabel('p')
ax2.set_ylabel('n')
## AC
CS3 = ax3.contour(X,Y,width.loc[(alpha, "AC")])
ax3.clabel(CS3, fontsize=10)
ax3.set_title('Agresti-Coull estimates')
ax3.set_xlabel('p')
ax3.set_ylabel('n')

plt.tight_layout()
display()
# -

# ### Contour plot for Estimated Relative Width of Confidence Interval

# +
#relative width
#alpha is 0.8
alpha = 0.8
fig0 = plt.figure()
fig0.suptitle('Relative Confidence Width for different methods with respect to \
Clopper-Pearson interval with Alpha=0.8')
ax0 = fig0.add_subplot(2, 2, 1)
ax2 = fig0.add_subplot(2, 2, 2) 
ax3 = fig0.add_subplot(2, 2, 3) 
## Normal
CS0 = ax0.contour(X,Y,width.loc[(alpha, "Normal")]/width.loc[(alpha, "CP")])
ax0.clabel(CS0, fontsize=10)
ax0.set_title('Normal Approximation')
ax0.set_xlabel('p')
ax0.set_ylabel('n')
## Jeffrey
CS2 = ax2.contour(X,Y,width.loc[(alpha, "Jeffrey")]/width.loc[(alpha, "CP")])
ax2.clabel(CS2, fontsize=10)
ax2.set_title('Jeffrey\'s method')
ax2.set_xlabel('p')
ax2.set_ylabel('n')
## AC
CS3 = ax3.contour(X,Y,width.loc[(alpha, "AC")]/width.loc[(alpha, "CP")])
ax3.clabel(CS3, fontsize=10)
ax3.set_title('Agresti-Coull estimates')
ax3.set_xlabel('p')
ax3.set_ylabel('n')

plt.tight_layout()
display()

# +
#relative width
#alpha is 0.9
alpha = 0.9
fig0 = plt.figure()
fig0.suptitle('Relative Confidence Width for different methods with respect to \
Clopper-Pearson interval with Alpha=0.9')
ax0 = fig0.add_subplot(2, 2, 1)
ax2 = fig0.add_subplot(2, 2, 2) 
ax3 = fig0.add_subplot(2, 2, 3) 
## Normal
CS0 = ax0.contour(X,Y,width.loc[(alpha, "Normal")]/width.loc[(alpha, "CP")])
ax0.clabel(CS0, fontsize=10)
ax0.set_title('Normal Approximation')
ax0.set_xlabel('p')
ax0.set_ylabel('n')
## Jeffrey
CS2 = ax2.contour(X,Y,width.loc[(alpha, "Jeffrey")]/width.loc[(alpha, "CP")])
ax2.clabel(CS2, fontsize=10)
ax2.set_title('Jeffrey\'s method')
ax2.set_xlabel('p')
ax2.set_ylabel('n')
## AC
CS3 = ax3.contour(X,Y,width.loc[(alpha, "AC")]/width.loc[(alpha, "CP")])
ax3.clabel(CS3, fontsize=10)
ax3.set_title('Agresti-Coull estimates')
ax3.set_xlabel('p')
ax3.set_ylabel('n')

plt.tight_layout()
display()

# +
#relative width
#alpha is 0.95
alpha = 0.95
fig0 = plt.figure()
fig0.suptitle('Relative Confidence Width for different methods with respect to \
Clopper-Pearson interval with Alpha=0.95')
ax0 = fig0.add_subplot(2, 2, 1)
ax2 = fig0.add_subplot(2, 2, 2) 
ax3 = fig0.add_subplot(2, 2, 3) 
## Normal
CS0 = ax0.contour(X,Y,width.loc[(alpha, "Normal")]/width.loc[(alpha, "CP")])
ax0.clabel(CS0, fontsize=10)
ax0.set_title('Normal Approximation')
ax0.set_xlabel('p')
ax0.set_ylabel('n')
## Jeffrey
CS2 = ax2.contour(X,Y,width.loc[(alpha, "Jeffrey")]/width.loc[(alpha, "CP")])
ax2.clabel(CS2, fontsize=10)
ax2.set_title('Jeffrey\'s method')
ax2.set_xlabel('p')
ax2.set_ylabel('n')
## AC
CS3 = ax3.contour(X,Y,width.loc[(alpha, "AC")]/width.loc[(alpha, "CP")])
ax3.clabel(CS3, fontsize=10)
ax3.set_title('Agresti-Coull estimates')
ax3.set_xlabel('p')
ax3.set_ylabel('n')

plt.tight_layout()
display()
