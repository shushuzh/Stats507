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
#from collections import defaultdict
from IPython.core.display import display, HTML
import math
import matplotlib.pyplot as plt 
import matplotlib as mpl
# 79: -------------------------------------------------------------------------

# # Question 0
#
# ## Data Files
#  - RECS microdata file for 2009 is [here](https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public.csv);
#  - Weight file for RECS microdata in 2009 is [here](https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public_repweights.csv);
#  - RECS microdata file for 2015 is [here](https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv).

mic_2009 = pd.read_csv(\
    "https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public.csv")
weight_2009 = pd.read_csv(\
    "https://www.eia.gov/consumption/residential/data/2009/csv/recs2009_public_repweights.csv")
mic_2015 = pd.read_csv(\
    "https://www.eia.gov/consumption/residential/data/2015/csv/recs2015_public_v4.csv")

# ## Variables

#Basic Variables (excluding replicated weights)
variables = [
"DOEID",  #unit ID
"REGIONC", #Census region
"NWEIGHT", #final sample weight
"HDD65", #heating degree days
"CDD65" #cooling degree days
#"BRRWT.*" #weights
]

#2009
remain_2009 = mic_2009[variables]
weight_2009 = weight_2009.drop(columns="NWEIGHT")
df_09 = pd.merge(remain_2009, weight_2009, on="DOEID")
print(df_09.head())
#2015
regex_stm = r"DOEID|REGIONC|NWEIGHT|HDD65|CDD65|BRRWT.*"
df_15 = mic_2015.filter(regex=regex_stm, axis=1)
print(df_15.head())
#Export data
df_09.to_pickle('mic_09.pkl')
df_09.to_pickle('mic_15.pkl')

# ## Weights and Replicate Weights
# The instruction for using replicate weights is [here](https://www.eia.gov/consumption/residential/data/2015/pdf/microdata_v3.pdf) (Microdata, 2015). 
#
# The estimate of standard error (for weighted point estimate) is 
# $$
# \sqrt{\frac{1}{R(1-\epsilon)^2}\sum_{r=1}^R (\hat{\theta_r}-\hat{\theta})^2}
# $$
# where $\theta$ is a population parameter of interest; $\hat{\theta}$ is the estimate from the full sample for $\theta$; $\hat{\theta_r}$ is the estimate from the r-th replicate subsample by using replicate weights; and $\epsilon$ be the Fay coefficient, $0 \leq \epsilon <1$. For both of the dataset, $\epsilon = 0.5$.

# # Question 1

# ## (a)
# I keep the variables for "unit ID, Census region, final sample weight, heating degree days, cooling degree days" for both datasets. And I change "unit ID" and "census region" to categorical variables.  

# Change the variable names to be more literate
remain_2009 = remain_2009.rename(columns={
                            "DOEID":"ids",  #unit ID
                            "REGIONC":"region", #Census region
                            "NWEIGHT":"weight", #final sample weight
                            "HDD65":"heat", #heating degree days
                            "CDD65":"cool" #cooling degree days
                            })
# convert the data type to proper ones
remain_2009 = remain_2009.convert_dtypes()
# change "unit ID" and "census region" to categorical variables
for var in ['ids', 'region']:
    remain_2009[var] = pd.Categorical(remain_2009[var]) 
print(remain_2009.dtypes)
print(remain_2009.head())

# Change the variable names to be more literate
remain_2015 = mic_2015[variables]
remain_2015 = remain_2015.rename(columns={
                            "DOEID":"ids",  #unit ID
                            "REGIONC":"region", #Census region
                            "NWEIGHT":"weight", #final sample weight
                            "HDD65":"heat", #heating degree days
                            "CDD65":"cool" #cooling degree days
                            })
# convert the data type to proper ones
remain_2015 = remain_2015.convert_dtypes()
# change "unit ID" and "census region" to categorical variables
for var in ['ids', 'region']:
    remain_2015[var] = pd.Categorical(remain_2015[var]) 
print(remain_2015.dtypes)
print(remain_2015.head())

# ## (b)
#
# ### Long Format for 2009 Data

weight_2009 = weight_2009.rename(columns={"DOEID":"ids"})
weight_2009.columns

weight_2009_long = pd.melt(weight_2009, 
                           id_vars=["ids"], 
                           value_vars=weight_2009.filter(like= 'brr_weight_'))
print(weight_2009_long)

# ### Long Format for 2015 Data

regex_stm = r"DOEID|BRRWT.*"
weight_2015 = df_15.filter(regex=regex_stm, axis=1)
weight_2015 = weight_2015.rename(columns={"DOEID":"ids"})
weight_2015.head()

weight_2015_long = pd.melt(weight_2015, 
                           id_vars=["ids"], 
                           value_vars=weight_2015.filter(like= 'BRRWT'))
print(weight_2015_long)

# # Question 2

# ## (a)

com_2009 = pd.merge(remain_2009,weight_2009, on="ids")
#print(com_2009.head())
com_2015 = pd.merge(remain_2015,weight_2015, on="ids")
#print(com_2015.head())

def point(weight,theta):
    """
    Calculate the weighted point estimate. 
    
    Parameters
    ----------
    weight : sample weights;
    theta: data of which the mean is to be estimated.
    
    Returns
    -------
    Weighted point estimator of the data.
    """
    return ((weight*theta).sum()/weight.sum())
# + "Weighted estimate of cooling degree days:" + str(point_cool))
def est(df):
    """
    Calculate the weighted point estimate and interval estimate for both heating and cooling days of the dataset. 
    
    Parameters
    ----------
    df : Dataset.
    
    Returns
    -------
    (point estimate for heating, 
    interval estimate for heating,
    point estimate for cooling, 
    interval estimate for cooling,
    standard deviation for heating,
    standard deviation for cooling)
    """
    #Point estimate of Heating Days
    point_heat = point(df['weight'],df['heat'])
    #Point estimate of Cooling Days
    point_cool = point(df['weight'],df['cool'])
    #Interval estimate of Heating Days
    R = df.shape[1]-5
    epsilon = 0.5
    diff = 0
    for i in range(R):
        theta_i_heat = point(df.iloc[:,i+5],df['heat'])
        diff +=(theta_i_heat-point_heat)**2
    sd_heat = math.sqrt(diff/(R*(1-epsilon)**2))
    inter_heat = (point_heat - 1.96*sd_heat, point_heat + 1.96*sd_heat)
    #Interval estimate of Cooling Days
    diff = 0
    for i in range(R):
        theta_i_cool = point(df.iloc[:,i+5],df['cool'])
        diff +=(theta_i_cool-point_cool)**2
    sd_cool = math.sqrt(diff/(R*(1-epsilon)**2))
    inter_cool = (point_cool - 1.96*sd_cool, point_cool + 1.96*sd_cool)
    
    return (round(point_heat),(round(inter_heat[0]),round(inter_heat[1])),
            #point and interval estimate for heating
            round(point_cool),(round(inter_cool[0]),round(inter_cool[1])),
            #point and interval estimate for cooling
            round(sd_heat),round(sd_cool))#sd


# -

est_2009 = (
com_2009
.groupby('region')
.apply(est)
)
est_2009

df_2009 = pd.DataFrame(tuple(est_2009), index=['1','2','3','4'])
print(df_2009)
table_2009 = df_2009.iloc[:,0:4]
table_2009.columns = [
                    "Point Estimate for Heating Days", 
                    "Interval Estimate for Heating Days", 
                    "Point Estimate for Cooling Days", 
                    "Interval Estimate for Cooling Days"]
table_2009.index.name = "region"
#print(table_2009)

# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 1.</b> <em> Point and Interval Estimate for Heating and Cooling Degree Days in 2009.</em>
"""
t1 = table_2009.to_html()
t1 = t1.rsplit('\n')
t1.insert(1, cap)
tab1 = ''
for i, line in enumerate(t1):
    tab1 += line
    if i < (len(t1) - 1):
        tab1 += '\n'
display(HTML(tab1))

est_2015 = (
com_2015
.groupby('region')
.apply(est)
)
est_2015

df_2015 = pd.DataFrame(tuple(est_2015), index = ['1','2','3','4'])
table_2015 = df_2015.iloc[:,0:4]
table_2015.columns = [
                    "Point Estimate for Heating Days", 
                    "Interval Estimate for Heating Days", 
                    "Point Estimate for Cooling Days", 
                    "Interval Estimate for Cooling Days"]
table_2015.index.name = "region"
#print(table_2015)

# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 2.</b> <em> Point and Interval Estimate for Heating and Cooling Degree Days in 2015.</em>
"""
t2 = table_2015.to_html()
t2 = t2.rsplit('\n')
t2.insert(1, cap)
tab2 = ''
for i, line in enumerate(t2):
    tab2 += line
    if i < (len(t2) - 1):
        tab2 += '\n'
display(HTML(tab2))

# ## (b)

# +
# Calculate point and interval estimate for difference: -----------------------
point_diff_heat = est_2015.apply(lambda t, x: t[x], args=(0,)) \
- est_2009.apply(lambda t, x: t[x], args=(0,))

sd_diff_heat = (est_2009.apply(lambda t, x: x**2, args=(4,)) \
                + est_2015.apply(lambda t, x: x**2, args=(4,)))\
.apply(lambda x: math.sqrt(x))

point_diff_cool = est_2015.apply(lambda t, x: t[x], args=(2,)) \
- est_2009.apply(lambda t, x: t[x], args=(2,))

sd_diff_cool = (est_2009.apply(lambda t, x: x**2, args=(5,)) \
                + est_2015.apply(lambda t, x: x**2, args=(5,)))\
.apply(lambda x: math.sqrt(x))

difference = {"Point Estimate for Difference of Heating": \
              list(point_diff_heat), 
              "Lower Interval for Difference of Heating": \
              list(round(point_diff_heat - 1.96*sd_diff_heat)), 
              "Upper Interval for Difference of Heating": \
              list(round(point_diff_heat + 1.96*sd_diff_heat)),
              "Point Estimate for Difference of Cooling": \
              list(point_diff_cool), 
              "Lower Interval for Difference of Cooling": \
              list(round(point_diff_cool - 1.96*sd_diff_cool)), 
              "Upper Interval for Difference of Cooling": \
              list(round(point_diff_cool + 1.96*sd_diff_cool,0))}
difference = pd.DataFrame(data = difference, index = ['1','2','3','4'])
difference.index.name = "region"
# -

# construct a table, include a caption: ---------------------------------------
cap = """
<b> Table 3.</b> <em> Point and Interval Estimate for Difference of Heating 
and Cooling Degree Days between 2009 and 2015. (data of 2015 - data of 2009)</em>
"""
t3 = difference.to_html()
t3 = t3.rsplit('\n')
t3.insert(1, cap)
tab3 = ''
for i, line in enumerate(t3):
    tab3 += line
    if i < (len(t3) - 1):
        tab3 += '\n'
display(HTML(tab3))

# ## Question 3
#
# ### Error Bar for Heating and Cooling Days of Micro Data in 2009

fig1 = plt.figure() 
fig1.suptitle('Error Bar for Heating and Cooling Days of Micro Data in 2009', 
              fontsize=14, 
              fontweight='bold')
ax1 = fig1.add_subplot()
plt.errorbar(
    x=df_2009.index, 
    y=df_2009[0], 
    yerr=df_2009[4],
    fmt='.r')
plt.errorbar(
    x=df_2009.index, 
    y=df_2009[2], 
    yerr=df_2009[5],
    fmt='.b')
plt.legend(["Heating","Cooling"])
ax1.set_xlabel('Region')
ax1.set_ylabel('Degree Days')
plt.show()

# ### Error Bar for Heating and Cooling Days of Micro Data in 2015

fig2 = plt.figure() 
fig2.suptitle('Error Bar for Heating and Cooling Days of Micro Data in 2015', 
              fontsize=14, 
              fontweight='bold')
ax2 = fig2.add_subplot()
plt.errorbar(
    x=df_2015.index, 
    y=df_2015[0], 
    yerr=df_2015[4],
    fmt='.r')
plt.errorbar(
    x=df_2015.index, 
    y=df_2015[2], 
    yerr=df_2015[5],
    fmt='.b')
plt.legend(["Heating","Cooling"])
ax2.set_xlabel('Region')
ax2.set_ylabel('Degree Days')
plt.show()

# ### Error Bar for Difference between 2009 and 2015 for Heating and Cooling Days of Micro Data (subtraction of 2015 and 2009)

fig3 = plt.figure() 
fig3.suptitle('Error Bar for Difference between 2009 and 2015 \
for Heating and Cooling Days of Micro Data', 
              fontsize=14, 
              fontweight='bold')
ax3 = fig3.add_subplot()
plt.errorbar(
    x=difference.index, 
    y=difference.iloc[:,0], 
    yerr=difference.iloc[:,2] - difference.iloc[:,0],
    fmt='.r')
plt.errorbar(
    x=difference.index, 
    y=difference.iloc[:,3], 
    yerr=difference.iloc[:,5] - difference.iloc[:,3],
    fmt='.b')
plt.legend(["Heating","Cooling"], loc='lower left')
ax3.set_xlabel('Region')
ax3.set_ylabel('Degree Days')
plt.show()

# # References
# Microdata, 2015. Residential Energy Consumption Survey (RECS). US Department of Energy: Washington, DC, USA.
