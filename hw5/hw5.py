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

# # Importing

# modules: --------------------------------------------------------------------
import numpy as np
import pandas as pd
from timeit import Timer
from os.path import exists
from collections import defaultdict
from IPython.core.display import display, HTML
import statistics
from scipy.stats import t, bootstrap
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import statsmodels.api as sm
import statsmodels.formula.api as smf
from warnings import warn
# 79: -------------------------------------------------------------------------

# # Question 0

# read tooth growth data
file = 'tooth_growth.feather'
if exists(file):
    tg_data = pd.read_feather(file)
else: 
    tooth_growth = sm.datasets.get_rdataset('ToothGrowth')
    #print(tooth_growth.__doc__)
    tg_data = tooth_growth.data
    tg_data.to_feather(file)


# log transform the tooth length
tg_data['log_len'] = np.log(tg_data['len'])
# change dose to categorical variable
tg_data['dose'] = pd.Categorical(tg_data['dose'])
mod1 = smf.ols('log_len ~ supp * dose', data=tg_data) 
res1 = mod1.fit()
res1.summary2()

# We compute $R^2$ and adjusted $R^2$ using the following equations:
# $$
# R^2 = \frac{||\hat{y} - \bar{y}||^2}{||y-\bar{y}||^2},
# $$
# $$
# Adjusted~R^2 = 1 - (1-R^2)\frac{n-1}{n-p-1}.
# $$
#
# The handcrafted $R^2$ and adjusted $R^2$ are exactly the same as in the result object. 

# Compute R^2
y_hat = res1.predict()
R2 = statistics.variance(y_hat)/statistics.variance(tg_data['log_len'])
n = tg_data.shape[0]
p = tg_data['supp'].nunique()-1 + tg_data['dose'].nunique()-1 \
+ (tg_data['supp'].nunique()-1)*(tg_data['dose'].nunique()-1)
adj_R2 = 1-(1-R2)*(n-1)/(n-p-1)
adj_R2

# # Question 1
# In this question, I first convert the all the "tc_" variables (i.e. "tc_01" to "tc_32") to binary variables with "Permanent tooth present" to be 1 and all the others to be 0. Then, I calculate the mean of each tooth count for each age group. Then, I use minimum age, peak age, 20 (which is larger than the largest minimum age among all the teeth), 40, 60 as the knots for the spline. Note here, the minimum age is defined as the minimum age of the mean>0.1, the reason being that if minimum age is too small, the design matrix is likely to be singular. The peak age is defined as the age of largest mean. Finally, I fit logistic regression for each of variables with B spline of the aforementioned knotes and degree 3. 
#
# (Instead of picking one variable at (a), I fit models to every variable.)

# Import data file: ------------------------------------------------------
path = './'
demo_file = path + '/demo.feather'
ohx_file = path + '/ohx.feather'
demo = pd.read_feather(demo_file)
demo_less = demo[["id","age"]]
ohx = pd.read_feather(ohx_file)
ohx_less = ohx.filter(regex=r"id|^tc", axis=1)
#merge data
data = pd.merge(demo_less,ohx_less,on='id',how='left')
data
# limit the analysis to age>=12
#data = data[data['age']>=12]

column_names = data.columns[2:]
# convert tc variables to binary variables
data.iloc[:,2:] = np.where(data.iloc[:,2:]=='Permanent tooth present',1,0)
# compute the average status for each group of age
ave = data.groupby('age').mean().iloc[:,1:]
data

#for each tooth
fig, ax = plt.subplots(nrows=8,ncols=4,sharex=True,sharey=True)
fig.set_size_inches(16,24)
i = 0
for name in column_names:
    #print(name)
    average = ave[name]
    ## find minimum age of the tooth appears
    epsilon = 0.1
    k0 = average[average>epsilon].index[0]
    #print(k0)
    ## find the peak age
    k1 = np.argmax(average)
    #print(k1)
    knots = [k0+1,k1,20,40,60]
    #knots[knots>=k0]
    knots.sort()
    mod2 = smf.logit(str(name)+'~bs(age,knots=knots,degree=3)',data=data)
    res2 = mod2.fit(disp=False)
    res2.summary()
    data[str(name)+'_hat'] = mod2.predict(params=res2.params)
    r = i // 8
    c = i % 8
    (data
     .groupby('age')[[name,str(name)+'_hat']]
     .mean()
     .plot
     .line(ax=ax[r,c])
    )
    i += 1 

# # Question 2
# I choose the first tooth to do the analysis. As we can see from the figure, all of the dots approximately lie on the line, i.e., the expected and observed means are approximately equal. Therefore, my model is considered well-calibrated. 

#split the data into deciles
data['tc_01_decile'] = pd.qcut(data['tc_01_hat'], 10, labels=False)

decile = (data
 .groupby('tc_01_decile')[['tc_01','tc_01_hat']]
 .mean()
)
x = np.linspace(0,0.35,100)
y = x
plt.plot(x, y, '-r', label='y=x')
plt.scatter(decile['tc_01'],decile['tc_01_hat'])
plt.title('Observed and Expected Proportion for tc_01')
plt.xlabel('Observed', color='#1C2833')
plt.ylabel('Expected', color='#1C2833')
plt.legend(loc='upper left')
plt.grid()
plt.show()
