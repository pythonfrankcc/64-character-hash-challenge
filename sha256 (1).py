
# coding: utf-8

# In[46]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[47]:


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from scipy.sparse import *


# In[48]:


#readintg the dataset
df = pd.read_csv("../input/hash-code-dataset/100k.csv")


# In[49]:


df.head()


# In[50]:


#since the hashing yields an intermediate result we can see if that result is repeated anywhere and use that to train our data if so
print("The number of unique intermediary results is: ",df['result'].nunique())
print("The number of unique hashes is: ",df['hash'].nunique())
print("The number of actual elements in the df[result] column is: ",df['result'].count())


# In[51]:


#finding the missing values in the dataset and sorting them in order
df.isnull().sum().sort_values(ascending = False)


# In[52]:


#creating an additional column thatgets the length of each hash to see whether there is a correlation with the results column in the future
df['hash_length'] = df['hash'].str.len()
df.head()


# In[53]:


#break down each of the hashed strings to see the components but for now let me just look at one
print(df['hash'][0])


# In[54]:


#from what we saw earlier we found out that there are a bunch of values that are duplicated in the results column
#selecting the duplicated rows
duplicate = df[df.duplicated('result')]


# In[55]:


print(duplicate)


# In[56]:


#from what we have in the duplicated we get a concept of what is called collission which is 
#multiple inputs can hash to the same output as is the case here
"""the game result that we have as our value for the result column is only related to the sha256 crypto
by the number derivation function which is done over and over again, so whatever we will find when we ran through the 
model is most likely the number derivation function relationship between the hash 256 and the result

You are supposed to find the relationsgip between the strings for you are supposed to generate a 64 bit string
that has not been seeing

sth that I learnt yesterday is that hasing functions are always deteerministic which means that the same input 
should always yield the same output even when done over and over again

To decipher this we have to start with what is a sha256 algos it is typically a 32 byte signature expressed as a 
64 characters strings"""


# In[57]:


# I want to sample a few hashes and see what they contain ehn they are broken down
print("This is the first hashed output: ",df['hash'][0])
print("This is the hundredth hashed output: ",df['hash'][100])
print("This is the second hundredth hashed output: ",df['hash'][200])
print("This is the third hundredth hashed output: ",df['hash'][300])
print("This is the 77th hashed output: ",df['hash'][77])


# In[58]:


#count columns for the hexadecimals
zeros_count = []
ones_count = []
two_count = []
three_count = []
four_count = []
five_count = []
six_count = []
seven_count = []
eight_count = []
nine_count = []
a_count = []
b_count = []
c_count = []
d_count = []
e_count = []
f_count = []
#function for the count
for i in df['hash']:
    zeros_counter = i.count('0')
    zeros_count.append(zeros_counter)
    ones_counter = i.count('1')
    ones_count.append(ones_counter)
    two_counter = i.count('2')
    two_count.append(two_counter)
    three_counter = i.count('3')
    three_count.append(three_counter)
    four_counter = i.count('4')
    four_count.append(four_counter)
    five_counter = i.count('5')
    five_count.append(five_counter)
    six_counter = i.count('6')
    six_count.append(six_counter)
    seven_counter = i.count('7')
    seven_count.append(seven_counter)
    eight_counter = i.count('8')
    eight_count.append(eight_counter)
    nine_counter = i.count('9')
    nine_count.append(nine_counter)
    a_counter = i.count('a')
    a_count.append(a_counter)
    b_counter = i.count('b')
    b_count.append(b_counter)
    c_counter = i.count('c')
    c_count.append(c_counter)
    d_counter = i.count('d')
    d_count.append(d_counter)
    e_counter = i.count('e')
    e_count.append(e_counter)
    f_counter = i.count('f')
    f_count.append(f_counter)

print(len(five_count))


# In[59]:


df['zeros_count'] = pd.DataFrame (zeros_count)


# In[60]:


df['ones_count'] = pd.DataFrame (ones_count)
df['two_count'] = pd.DataFrame (two_count)
df['three_count'] = pd.DataFrame (three_count)
df['four_count'] = pd.DataFrame (four_count)
df['five_count'] = pd.DataFrame (five_count)
df['six_count'] = pd.DataFrame (six_count)
df['seven_count'] = pd.DataFrame (seven_count)
df['eight_count'] = pd.DataFrame (eight_count)
df['nine_count'] = pd.DataFrame (nine_count)
df['a_count'] = pd.DataFrame (a_count)
df['b_count'] = pd.DataFrame (b_count)
df['c_count'] = pd.DataFrame (c_count)
df['d_count'] = pd.DataFrame (d_count)
df['e_count'] = pd.DataFrame (e_count)
df['f_count'] = pd.DataFrame (f_count)


# In[61]:


df.head()


# In[62]:


print(df['three_count'].shape)
print(df['hash'].shape)


# this ascertaines that the new columns and the has are of the same shapd and dimensionality

# In[63]:


#not to lose the data that is in the hex column I would rather convert the column into an int that is workable
#with our model rather than converting it into a label encoded thing
hash_signed_int = []
def signed_int(h):
    x = int(h, 16)
    return x
for i in df['hash']:
    a = signed_int(i)
    hash_signed_int.append(a)


# In[64]:


df['hash_signed_int'] = pd.DataFrame (hash_signed_int)


# In[65]:


df.dtypes

