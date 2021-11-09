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

# # Hierarchical indexing (MultiIndex)
# - Shushu Zhang
# - shushuz@umich.edu
# - Reference is [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html) 
#
# Hierarchical / Multi-level indexing is very exciting as it opens the door to some quite sophisticated data analysis and manipulation, especially for working with higher dimensional data. In essence, it enables you to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures like Series (1d) and DataFrame (2d).

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
