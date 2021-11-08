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

import numpy as np
import time
#from timeit import Timer
import pandas as pd
#from itables import show
from tabulate import tabulate
from scipy import stats


# # Question 0

# deal with second row; 69 characters.

# This is question 0 for [problem set 1](https://jbhender.github.io/Stats507/F21/ps/ps1.html) of [Stats 507](https://jbhender.github.io/Stats507/F21/).
#
# ***Question 0 is about Markdown.***
#
# The next question is about the **Fibonnaci sequence**, Fn=Fn−2+Fn−1. In part **a** we will define a Python function fib_rec().
#
# Below is a …
#
# ### Level 3 Header
# Next, we can make a bulleted list:
#
# - Item 1
#     - detail 1
#     - detail 2
# - Item 2
#
# Finally, we can make an enumerated list:
#
# a. Item 1 \
# b. Item 2 \
# c. Item 3
#

# # Question 2

# ## (a) fib_rec()

# +
# Define function
def fib_rec(n,a,b):
    """
    It is a recursive function that calculate Fibonnaci Sequence with input n and
    starting values, and returns the value of F_n. 
    
    Parameters
    ----------
    n : number in sequence;
    a : starting point F_0;
    b : starting point F_1. 
    
    Returns
    -------
    F_n.
    """
    if n==0:
        return (a)
    elif n==1:
        return (b)
    else:
        return (fib_rec(n-1,a,b)+fib_rec(n-2,a,b))

# Test
print(fib_rec(7,0,1))
print(fib_rec(11,0,1))
print(fib_rec(13,0,1))


# -

# ## (b) fib_for()

# +
# Define function
def fib_for(n,a,b):
    """
    Calculate Fibonnaci Sequence with a single input n and returns the value of F_n by a for loop. 
    
    Parameters
    ----------
    n : number in sequence.
    
    Returns
    -------
    F_n.
    """
    if n==0:
        return (a)
    elif n==1:
        return (b)
    else:
        F0 = a
        F1 = b
        for i in range(n-1):
            F2 = F0 + F1
            F0 = F1
            F1 = F2
        return (F2)

# Test
print(fib_rec(7,0,1))
print(fib_rec(11,0,1))
print(fib_rec(13,0,1))


# -

# ## (c) fib_whl()

# +
# Define function
def fib_whl(n,a,b):
    """
    Calculate Fibonnaci Sequence with a single input n and returns the value of F_n by a while loop. 
    Parameters
    ----------
    n : number in sequence.
    Returns
    -------
    F_n.
    """
    if n==0:
        return (a)
    elif n==1:
        return (b)
    else:
        F0 = a
        F1 = b
        i = 1
        while i<n:
            F2 = F0 + F1
            F0 = F1
            F1 = F2
            i += 1
        return (F2)

# Test
print(fib_rec(7,0,1))
print(fib_rec(11,0,1))
print(fib_rec(13,0,1))


# -

# ## (d) fib_rnd()

# +
# Define function
def fib_rnd(n,a,b):
    """
    Calculate Fibonnaci Sequence with a single input n and returns 
    the value of F_n using rounding method. 
    
    Parameters
    ----------
    n : number in sequence.
    
    Returns
    -------
    F_n.
    """
    phi = (1+np.sqrt(5))/2
    return (round(phi**n/np.sqrt(5))+a+b-1)

# Test
print(fib_rec(7,0,1))
print(fib_rec(11,0,1))
print(fib_rec(13,0,1))


# -

# ## (e) fib_flr()

# +
# Define function
def fib_flr(n,a,b):
    """
    Calculate Fibonnaci Sequence with a single input n 
    and returns the value of F_n using truncation method. 
    
    Parameters
    ----------
    n : number in sequence.
    Returns
    -------
    F_n.
    """
    phi = (1+np.sqrt(5))/2
    return (int(np.floor(phi**n/np.sqrt(5)+1/2))+a+b-1)

# Test
print(fib_rec(7,0,1))
print(fib_rec(11,0,1))
print(fib_rec(13,0,1))
# -

# ## (f) Comparison 
# In this section, I will show a nicely formatted table indicating the time each function will run for a particular number of Fibonnaci Sequence. 

fun_names = ['fib_rec','fib_for','fib_whl','fib_rnd','fib_flr']
n_ranges = range(5,40,5)
df = pd.DataFrame(columns = fun_names, index = n_ranges)

for fun in fun_names:
    for n in n_ranges:
        t0 = time.time()
        eval(fun)(n,0,1)
        t1 = time.time()
        df.at[n,fun] = t1-t0
print(df)     

# As we can see from both the detailed table above and the median summary table below, fib_rec function computes with far more time than the rest, indicating that using recursive function is less efficient than the others in terms of computational costs. 

print(tabulate(pd.DataFrame(df.median(axis = 0)),headers = ['Function','Median Computing Time']))


# # Question 2 - Pascal’s Triangle

# ## (a) Write a function to compute a specified row of Pascal’s triangle

def Pascal(n):
    """
    Compute a specified (n-th) row of Pascal’s triangle. 
    
    Parameters
    ----------
    n : number of the specified row.
    
    Returns
    -------
    n-th row of Pascal’s triangle.
    """
    P_n = [1]
    if n>0:
        for k in range(1,n+1):
            P_n.append(int(P_n[-1]*(n+1-k)/k))
    return (P_n)


#Test
Pascal(6)


# ## (b) Write a function for printing the first n rows of Pascal’s triangle using the conventional spacing with the numbers in each row staggered relative to adjacent rows.

def Pascal_firstn(n):
    """
    Compute the first n rows (until n-th row) of Pascal’s triangle. 
    
    Parameters
    ----------
    n : number of rows printed.
    
    Returns
    -------
    First n rows of Pascal’s triangle.
    """
    for i in range(n+1):
        print(" "*2*(n-i),end="")
        for j in range(i+1):
            print(str(Pascal(i)[j]).center(4), end="")
        print()    


# Test
Pascal_firstn(10)


# # Question 3 - Statistics 101

# ## (a)

def CI(data,level,ci_format = "string"):
    """
    Compute the confidence interval. 
    
    Parameters
    ----------
    data : a 1d Numpy array or any object coercable to such an array using np.array();
    level : confidence level;
    ci_format : (optional) default return a string with the format "\hat{\theta}[XX%CI:(\hat{\theta}_L,\hat{\theta}_U)]",
    if ci_format = None, return a dictionary with keys est, lwr, upr, and level.
    
    Returns
    -------
    depending on ci_format, the function returns either a string like 
    "\hat{\theta}[XX%CI:(\hat{\theta}_L,\hat{\theta}_U)]" or a dictionary with keys est, lwr, upr, and level.
    """
    data = np.asarray(data)
    if ((not isinstance(data, np.ndarray)) or data.ndim!=1):
        return ("Not passing a 1d Numpy array or anything ciercable to it.")
    est = np.mean(data)
    sd = np.std(data)
    z = stats.norm.ppf(level)
    lwr, upr = est - z*sd, est + z*sd
    dic = {'est':est, 'lwr':lwr, 'upr':upr, 'level':level}
    if ci_format==None:
        return (dic)
    else:
        return ("{est}[{level}%CI:({lwr},{upr})]".format_map(dic))
    


#Test
data = [1,2,3,4,5,6,7,8,9,10]
print(CI(data,0.95,ci_format = None))
print(CI(data,0.95))


# ## (b)

def CI_bin(data,level,method = "standard",ci_format = "string"):
    """
    Compute the confidence interval of binomial distribution using several methods. 
    
    Parameters
    ----------
    data : a 1d Numpy array or any object coercable to such an array using np.array();
    level : confidence level;
    method (optional): the method of computing the confidence interval, the values can be "standard" (default), "Clopper-Pearson", 
    "Jeffrey’s", and "Agresti-Coull";
    ci_format (optional): default return a string with the format "\hat{\theta}[XX%CI:(\hat{\theta}_L,\hat{\theta}_U)]",
    if ci_format = None, return a dictionary with keys est, lwr, upr, and level.
    
    Returns
    -------
    depending on method and ci_format, the function returns confidence interval either a string like 
    "\hat{\theta}[XX%CI:(\hat{\theta}_L,\hat{\theta}_U)]" or a dictionary with keys est, lwr, upr, and level.
    """
    data = np.asarray(data)
    if ((not isinstance(data, np.ndarray)) or data.ndim!=1):
        return ("Not passing a 1d Numpy array or anything ciercable to it.")
    est = np.mean(data)
    sd = np.std(data)
    z = stats.norm.ppf(level)
    lwr, upr = est - z*sd, est + z*sd
    dic = {'est':est, 'lwr':lwr, 'upr':upr, 'level':level}
    if ci_format==None:
        return (dic)
    else:
        return ("{est}[{level}%CI:({lwr},{upr})]".format_map(dic))
    





# ## (c)
