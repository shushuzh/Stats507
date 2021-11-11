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

from IPython.display import Image

# # Question 0
# The repository can be found [here](https://github.com/shushuzh/Stats507.git).

# # Question 1

# ## Step 1
# I added and pushed the file in [here](https://github.com/shushuzh/Stats507/blob/master/hw6/ps2_q3.ipynb) under "hw6" folder. The commit history for this operation is [here](https://github.com/shushuzh/Stats507/commit/0ce40e75a0c17b3288ea81c5d4aa9e4a3568592a). 
#
# I included the screenshot of my command line here. 

Image(url='q1_command/step1.png')

# ## Step 2
# I added and pushed the "README.md" file in [here](https://github.com/shushuzh/Stats507/blob/master/README.md). I added this file with [this commit](https://github.com/shushuzh/Stats507/commit/bf71244502287d39b7c2fb076f10d9683bc08216). And I figured I put the "README.md" file in the wrong directory, thus moving it with [this commit](https://github.com/shushuzh/Stats507/commit/6f453eef843bbddcd1e9cf4b4844efadd958b61c). 
#
# I included the screenshot of my command line here. 

Image(url='q1_command/step2.png')

# ## Step 4&5
# (This one is quite new to me so I am going to report in detail.) I mainly typed the following commands in the terminal:
# - git branch ps4\
# (Create a new branch called "ps4".)
# - git checkout ps4\
# (Switch to branch "ps4".)
# - git add ps2_q3.ipynb\
# (After I edit the file, I add the change to the branch "ps4".)
# - git commit -m "include gender in branch ps4"
# - git push -u origin ps4\
# (Set upstream branch ps4 and push the changes to that branch.)
# - git checkout master\
# (Switch to branch "master".)
# - git merge ps4\
# (Merge the commits of branch "ps4" to "master".)
#
# The link to the commit is [here](https://github.com/shushuzh/Stats507/commit/c273d68442fa7aeca522473d6047a6dc012394c7). 

Image(url='q1_command/step4_5(1).png')

Image(url='q1_command/step4_5(2).png')

# # Question 2
#
# For sub-question 2, the file is [here](https://github.com/shushuzh/Stats507/blob/master/pandas_notes/pd_topic_shushuz.py).

# # Question 3
# The commit for correction of ps1 question 0 is [here](https://github.com/shushuzh/Stats507/commit/7207f1abe3dfb59ea1c9a872ec03fa71582e4e44). 
