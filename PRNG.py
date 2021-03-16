
# coding: utf-8

# In[1]:


"""##Importing the libraries"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
from scipy.sparse import *


# In[2]:


from google.colab import drive
drive.mount('/content/drive')


# In[3]:


"""## Importing the dataset"""
dataset = pd.read_csv('/content/drive/MyDrive/PRNG/4-mil-test-set.csv')


# In[ ]:


#Understanding your dataset first 
dataset.head()
dataset.dtypes == 'object'


# In[ ]:


#separate the categorical variables from the numerical variables
num_vars = dataset.columns[dataset.dtypes != 'object']
cat_vars = dataset.columns[dataset.dtypes != 'object']


# In[ ]:


print(num_vars) 
print(cat_vars)


# In[ ]:


#finding the missing values in the dataset and sorting them in order
dataset.isnull().sum().sort_values(ascending = False)


# In[4]:


#dataset = dataset[dataset.result < 2000]
#dataset = dataset[dataset.id < 20000]
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values


# In[5]:


"""## Encoding categorical data"""
"""# Label Encoding the "hash" column"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


# In[6]:


"""## Splitting the dataset into the Training set and Test set"""

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[7]:


"""## Feature Scaling"""
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (1, 1000))
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[8]:


"""## Preparing the training set and the test set"""
X_train_df = pd.DataFrame(X_train)
y_train_df = pd.DataFrame(y_train)
X_train_df[2] = y_train_df[0]
training_set = np.array(X_train_df, dtype='int')


X_test_df = pd.DataFrame(X_test)
y_test_df = pd.DataFrame(y_test)
X_test_df[2] = y_test_df[0]
test_set = np.array(X_test_df, dtype='int')


# In[9]:


"""## Getting the number of ids and results"""

nb_id = int(max(max(training_set[:, 0], ), max(test_set[:, 0])))
nb_results = int(max(max(training_set[:, 1], ), max(test_set[:, 1])))


# In[10]:


"""## Converting the data into an array with id in lines and result in columns"""

def convert(data):
  new_data = []
  for id_users in range(1, nb_id + 1):
    id_result = data[:, 1] [data[:, 0] == id_users]
    id_hash = data[:, 2] [data[:, 0] == id_users]
    sha_hash = np.zeros(nb_results)
    sha_hash[id_result - 1] = id_hash
    new_data.append(list(sha_hash))
  return new_data
training_set = convert(training_set)
test_set = convert(test_set)


# In[11]:


"""## Converting the data into Torch tensors"""

training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)


# In[12]:


"""## Creating the architecture of the Neural Network"""

class SAE(nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nb_results, 30)
        self.fc2 = nn.Linear(30, 20)
        self.fc3 = nn.Linear(20, 10)
        self.fc4 = nn.Linear(10, 20)
        self.fc5 = nn.Linear(20, 30)
        self.fc6 = nn.Linear(30, nb_results)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.activation(self.fc4(x))
        x = self.activation(self.fc5(x))
        x = self.fc6(x)
        return x
sae = SAE()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)


# In[14]:


"""## Training the SAE"""

nb_epoch = 2000
for epoch in range(1, nb_epoch + 1):
  train_loss = 0
  s = 0.
  for id_user in range(nb_id):
    input = Variable(training_set[id_user]).unsqueeze(0)
    target = input.clone()
    if torch.sum(target.data > 0) > 0:
      output = sae(input)
      target.require_grad = False
      output[target == 0] = 0
      loss = criterion(output, target)
      mean_corrector = nb_results/float(torch.sum(target.data > 0) + 1e-10)
      loss.backward()
      train_loss += np.sqrt(loss.data*mean_corrector)
      s += 1.
      optimizer.step()
  print('epoch: '+str(epoch)+'loss: '+ str(train_loss/s))


# In[ ]:


"""## Testing the SAE"""

test_loss = 0
s = 0.
for id_user in range(nb_id):
  input = Variable(training_set[id_user]).unsqueeze(0)
  target = Variable(test_set[id_user]).unsqueeze(0)
  if torch.sum(target.data > 0) > 0:
    output = sae(input)
    target.require_grad = False
    output[target == 0] = 0
    loss = criterion(output, target)
    mean_corrector = nb_results/float(torch.sum(target.data > 0) + 1e-10)
    test_loss += np.sqrt(loss.data*mean_corrector)
    s += 1.
print('test loss: '+str(test_loss/s))


# In[ ]:


"""## Extracting the title of the id and result from the dataset"""
id_title = dataset.iloc[:nb_results, 0:1]
result_title = dataset.iloc[:nb_results, 1:-1]


# In[ ]:


"""## Choosing a User by the Test Set and taking the whole list of Hashes of that User"""
user_id = 101
user_hash = training_set.data.numpy()[user_id - 1, :].reshape(-1,1)
user_target = test_set.data.numpy()[user_id, :].reshape(-1,1)


# In[ ]:


"""## Making the Predictions using the Input"""
user_input = Variable(training_set[user_id]).unsqueeze(0)
predicted = sae(user_input)
predicted = predicted.data.numpy().reshape(-1,1)
predicted


# In[ ]:


"""## Combining all the info into one dataframe"""
hash_array = np.hstack([id_title, user_target, predicted])
hash_array = hash_array[hash_array[:, 1] > 0]
hash_df = pd.DataFrame(data=hash_array, columns=['id', 'Target Hash', 'Predicted'])
hash_df


# In[ ]:


dataset['result'].iloc[-1]

