#!/usr/bin/env python
# coding: utf-8

# # Selecting Subsets of Data 

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name) # name object 삭제

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Selecting Series data

# ### How to do it...

# In[2]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
city = college['CITY']
city


# In[3]:


city['Alabama A & M University']


# In[4]:


city.loc['Alabama A & M University']


# In[5]:


city.iloc[0]


# In[6]:


city[['Alabama A & M University', 'Alabama State University']]
city['Alabama A & M University', 'Alabama State University'] # 에러남
type(city[['Alabama A & M University', 'Alabama State University']]) # pandas.core.series.Series


# In[7]:


city.loc[['Alabama A & M University', 'Alabama State University']]


# In[8]:


city.iloc[[0, 4]]


# In[9]:


city['Alabama A & M University': 'Alabama State University']


# In[10]:


city[0:5]


# In[11]:


city.loc['Alabama A & M University': 'Alabama State University']


# In[12]:


city.iloc[0:5]


# In[13]:

# Boolean arrays
alabama_mask = city.isin(['Birmingham', 'Montgomery'])
type(alabama_mask)
city[alabama_mask]


# ### How it works...

# In[14]:


s = pd.Series([10, 20, 35, 28], index=[5,2,3,1])
s


# In[15]:


s[0:4]


# In[16]:


s[5]


# In[17]:


s[1]


# ### There's more...

# In[18]:


college.loc['Alabama A & M University', 'CITY']
type(college)
type(college.loc['Alabama A & M University', 'CITY'])



# In[19]:


college.iloc[0, 0]


# In[20]:


college.loc[['Alabama A & M University', 
             'Alabama State University'], 'CITY']


# In[21]:


college.iloc[[0, 4], 0]


# In[22]:


college.loc['Alabama A & M University':
            'Alabama State University', 'CITY']


# In[23]:


college.iloc[0:5, 0]


# In[24]:


city.loc['Reid State Technical College':
         'Alabama State University']


# ## Selecting DataFrame rows

# In[25]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.sample(5, random_state=42)


# In[26]:


college.iloc[60]


# In[27]:


college.loc['University of Alaska Anchorage']


# In[28]:


college.iloc[[60, 99, 3]]


# In[29]:


labels = ['University of Alaska Anchorage',
          'International Academy of Hair Design',
          'University of Alabama in Huntsville']
college.loc[labels]


# In[30]:


college.iloc[99:102]


# In[31]:


start = 'International Academy of Hair Design'
stop = 'Mesa Community College'
college.loc[start:stop]


# ### How it works...

# ### There's more...

# In[32]:


college.iloc[[60, 99, 3]].index.tolist()


# ## Selecting DataFrame rows and columns simultaneously

# ### How to do it...

# In[33]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.iloc[:3, :4]


# In[34]:


college.loc[:'Amridge University', :'MENONLY']


# In[35]:


college.iloc[:, [4,6]].head()


# In[36]:


college.loc[:, ['WOMENONLY', 'SATVRMID']].head()


# In[37]:


college.iloc[[100, 200], [7, 15]]


# In[38]:


rows = ['GateWay Community College',
        'American Baptist Seminary of the West']
columns = ['SATMTMID', 'UGDS_NHPI']
college.loc[rows, columns]


# In[39]:


college.iloc[5, -4]


# In[40]:


college.loc['The University of Alabama', 'PCTFLOAN']


# In[41]:


college.iloc[90:80:-2, 5]


# In[42]:


start = 'Empire Beauty School-Flagstaff'
stop = 'Arizona State University-Tempe'
college.loc[start:stop:-2, 'RELAFFIL']


# ### How it works...

# ### There's more...

# ## Selecting data with both integers and labels

# ### How to do it...

# In[43]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')


# In[44]:


col_start = college.columns.get_loc('UGDS_WHITE')
col_end = college.columns.get_loc('UGDS_UNKN') + 1
col_start, col_end


# In[45]:


college.iloc[:5, col_start:col_end]


# ### How it works...

# ### There's more...

# In[46]:


row_start = college.index[10]
row_end = college.index[15]
college.loc[row_start:row_end, 'UGDS_WHITE':'UGDS_UNKN']


# In[47]:


college.ix[10:16, 'UGDS_WHITE':'UGDS_UNKN']


# In[48]:


college.iloc[10:16].loc[:, 'UGDS_WHITE':'UGDS_UNKN']


# ## Slicing lexicographically

# ### How to do it...

# In[49]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')


# In[50]:


college.loc['Sp':'Su']


# In[51]:


college = college.sort_index()


# In[52]:


college.loc['Sp':'Su']

college.sort_index().loc['Sp':'Su']
# ### How it works...

# ### There's more...

# In[53]:


college = college.sort_index(ascending=False)
college.index.is_monotonic_decreasing


# In[54]:

college.loc['E':'B']

college.loc['A':'B'] # 아무것도 안들어옴
 
# In[ ]:




