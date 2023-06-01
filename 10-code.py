#!/usr/bin/env python
# coding: utf-8

# # Restructuring Data into a Tidy Form

import os
os.getcwd()
os.chdir('D:/pandas_cookbook')




# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Tidying variable values as column names with stack

# In[2]:


state_fruit = pd.read_csv('data/state_fruit.csv', index_col=0)
state_fruit


# ### How to do it...

# In[3]:


state_fruit.stack()

stf = state_fruit.stack().copy()
stf.shape
type(stf)

# In[4]:

# reset_index를 이용해 Data Frame으로 변환
(state_fruit
   .stack()
   .reset_index()
)


# In[5]:


(state_fruit
   .stack()
   .reset_index()
   .rename(columns={'level_0':'state', 
      'level_1': 'fruit', 0: 'weight'})
)


# In[6]:


(state_fruit
    .stack()
    .rename_axis(['state', 'fruit'])
)


# In[7]:


(state_fruit
    .stack()
    .rename_axis(['state', 'fruit'])
    .reset_index(name='weight')
)

# ?pd.reset_index
# help(pd.Series.reset_index)

# ### How it works...

# ### There's more...

# In[8]:


state_fruit2 = pd.read_csv('data/state_fruit2.csv')
state_fruit2


# In[9]:


state_fruit2.stack()


# In[10]:


state_fruit2.set_index('State').stack()


# ## Tidying variable values as column names with melt

# ### How to do it...

# In[11]:


state_fruit2 = pd.read_csv('data/state_fruit2.csv')
state_fruit2


# In[12]:


state_fruit2.melt(id_vars=['State'],
    value_vars=['Apple', 'Orange', 'Banana'])


# In[13]:


state_fruit2.melt(id_vars=['State'],
                   value_vars=['Apple', 'Orange', 'Banana'],
                   var_name='Fruit',
                   value_name='Weight')


# ### How it works...

# ### There's more...

# In[14]:


state_fruit2.melt()


# In[15]:


state_fruit2.melt(id_vars='State')


# ## Stacking multiple groups of variables simultaneously

# In[16]:


movie = pd.read_csv('data/movie.csv')
actor = movie[['movie_title', 'actor_1_name',
               'actor_2_name', 'actor_3_name',
               'actor_1_facebook_likes',
               'actor_2_facebook_likes',
               'actor_3_facebook_likes']]
actor.head()


# ### How to do it...

# In[17]:


def change_col_name(col_name):
    col_name = col_name.replace('_name', '')
    if 'facebook' in col_name:
        fb_idx = col_name.find('facebook')
        col_name = (col_name[:5] + col_name[fb_idx - 1:] 
               + col_name[5:fb_idx-1])
    return col_name


# In[18]:


actor2 = actor.rename(columns=change_col_name)
actor2


# In[19]:


stubs = ['actor', 'actor_facebook_likes']
actor2_tidy = pd.wide_to_long(actor2,
    stubnames=stubs,
    i=['movie_title'],
    j='actor_num',
    sep='_')
actor2_tidy.head()


# ### How it works...

# ### There's more...

# In[20]:


df = pd.read_csv('data/stackme.csv')
df


# In[21]:


df.rename(columns = {'a1':'group1_a1', 'b2':'group1_b2',
                     'd':'group2_a1', 'e':'group2_b2'})


# In[22]:


pd.wide_to_long(
       df.rename(columns = {'a1':'group1_a1', 
                 'b2':'group1_b2',
                 'd':'group2_a1', 'e':'group2_b2'}),
    stubnames=['group1', 'group2'],
    i=['State', 'Country', 'Test'],
    j='Label',
    suffix='.+',
    sep='_')


# ## Inverting stacked data

# ### How to do it...

# In[23]:


usecol_func = lambda x: 'UGDS_' in x or x == 'INSTNM'
college = pd.read_csv('data/college.csv',
    index_col='INSTNM',
    usecols=usecol_func)
college


# In[24]:


college_stacked = college.stack()
college_stacked


# In[25]:


college_stacked.unstack()


# In[26]:


college2 = pd.read_csv('data/college.csv',
   usecols=usecol_func)
college2


# In[27]:


college_melted = college2.melt(id_vars='INSTNM',
    var_name='Race',
    value_name='Percentage')
college_melted


# In[28]:


melted_inv = college_melted.pivot(index='INSTNM',
    columns='Race',
    values='Percentage')
melted_inv


# In[29]:


college2_replication = (melted_inv
    .loc[college2['INSTNM'], college2.columns[1:]]
    .reset_index()
)
college2.equals(college2_replication)


# ### How it works...

# ### There's more...

# In[30]:


college.stack().unstack(0)


# In[31]:


college.T
college.transpose()


# ## Unstacking after a groupby aggregation

# ### How to do it...

# In[32]:


employee = pd.read_csv('data/employee.csv')
(employee
    .groupby('RACE')
    ['BASE_SALARY']
    .mean()
    .astype(int)
)


# In[33]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
)


# In[34]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
    .unstack('GENDER')
)


# In[35]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .astype(int)
    .unstack('RACE')
)


# ### How it works...

# ### There's more...

# In[36]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY']
    .agg(['mean', 'max', 'min'])
    .astype(int)
)


# In[37]:


(employee
    .groupby(['RACE', 'GENDER'])
    ['BASE_SALARY']
    .agg(['mean', 'max', 'min'])
    .astype(int)
    .unstack('GENDER')
)


# ## Replicating pivot_table with a groupby aggregation

# ### How to do it...

# In[38]:


flights = pd.read_csv('data/flights.csv')
fpt = flights.pivot_table(index='AIRLINE',
    columns='ORG_AIR',
    values='CANCELLED',
    aggfunc='sum',
    fill_value=0).round(2)
fpt


# In[39]:


(flights
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['CANCELLED']
    .sum()
)


# In[40]:


fpg = (flights
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['CANCELLED']
    .sum()
    .unstack('ORG_AIR', fill_value=0)
)


# In[41]:


fpt.equals(fpg)


# ### How it works...

# ### There's more...

# In[42]:


flights.pivot_table(index=['AIRLINE', 'MONTH'],
    columns=['ORG_AIR', 'CANCELLED'],
    values=['DEP_DELAY', 'DIST'],
    aggfunc=['sum', 'mean'],
    fill_value=0)


# In[43]:


(flights
    .groupby(['AIRLINE', 'MONTH', 'ORG_AIR', 'CANCELLED']) 
    ['DEP_DELAY', 'DIST'] 
    .agg(['mean', 'sum']) 
    .unstack(['ORG_AIR', 'CANCELLED'], fill_value=0) 
    .swaplevel(0, 1, axis='columns')
)


# ## Renaming axis levels for easy reshaping

# ### How to do it...

# In[44]:


college = pd.read_csv('data/college.csv')
(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
)


# In[45]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
)


# In[46]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
)


# In[47]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .swaplevel('AGG_FUNCS', 'STABBR',
       axis='index')
)


# In[48]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .swaplevel('AGG_FUNCS', 'STABBR', axis='index') 
    .sort_index(level='RELAFFIL', axis='index') 
    .sort_index(level='AGG_COLS', axis='columns')
)


# In[49]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack('AGG_FUNCS')
    .unstack(['RELAFFIL', 'STABBR'])
)


# In[50]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .stack(['AGG_FUNCS', 'AGG_COLS'])
)


# In[51]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis(['AGG_COLS', 'AGG_FUNCS'], axis='columns')
    .unstack(['STABBR', 'RELAFFIL']) 
)


# ### How it works...

# ### There's more...

# In[52]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATMTMID'] 
    .agg(['size', 'min', 'max'])
    .rename_axis([None, None], axis='index') 
    .rename_axis([None, None], axis='columns')
)


# ## Tidying when multiple variables are stored as column names

# ### How to do it...

# In[53]:


weightlifting = pd.read_csv('data/weightlifting_men.csv')
weightlifting


# In[54]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)


# In[55]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
)


# In[56]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
)


# In[57]:


(weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
    .assign(Sex=lambda df_: df_.Sex.str[0])
)


# In[58]:


melted = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)
tidy = pd.concat([melted
           ['sex_age']
           .str.split(expand=True)
           .rename(columns={0:'Sex', 1:'Age Group'})
           .assign(Sex=lambda df_: df_.Sex.str[0]),
          melted[['Weight Category', 'Qual Total']]],
          axis='columns'
)
tidy


# In[59]:


melted = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
)
(melted
    ['sex_age']
    .str.split(expand=True)
    .rename(columns={0:'Sex', 1:'Age Group'})
    .assign(Sex=lambda df_: df_.Sex.str[0],
            Category=melted['Weight Category'],
            Total=melted['Qual Total'])
)


# ### How it works...

# ### There's more...

# In[60]:


tidy2 = (weightlifting
    .melt(id_vars='Weight Category',
          var_name='sex_age',
          value_name='Qual Total')
    .assign(Sex=lambda df_:df_.sex_age.str[0],
            **{'Age Group':(lambda df_: (df_
                .sex_age
                .str.extract(r'(\d{2}[-+](?:\d{2})?)',
                             expand=False)))})
    .drop(columns='sex_age')
)


# In[61]:


tidy2


# In[62]:


tidy.sort_index(axis=1).equals(tidy2.sort_index(axis=1))


# ## Tidying when multiple variables are stored is a single column

# ### How to do it...

# In[63]:


inspections = pd.read_csv('data/restaurant_inspections.csv',
    parse_dates=['Date'])
inspections


# In[64]:


inspections.pivot(index=['Name', 'Date'],
    columns='Info', values='Value')


# In[65]:


inspections.set_index(['Name','Date', 'Info'])


# In[66]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
)


# In[67]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
    .reset_index(col_level=-1)
)


# In[68]:


def flatten0(df_):
    df_.columns = df_.columns.droplevel(0).rename(None)
    return df_


# In[69]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .unstack('Info')
    .reset_index(col_level=-1)
    .pipe(flatten0)
)


# In[70]:


(inspections
    .set_index(['Name','Date', 'Info']) 
    .squeeze() 
    .unstack('Info') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ### How it works...

# ### There's more...

# In[71]:


(inspections
    .pivot_table(index=['Name', 'Date'],
                 columns='Info',
                 values='Value',
                 aggfunc='first') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ## Tidying when two or more values are stored in the same cell

# ### How to do it..

# In[72]:


cities = pd.read_csv('data/texas_cities.csv')
cities


# In[73]:


geolocations = cities.Geolocation.str.split(pat='. ',
    expand=True)
geolocations.columns = ['latitude', 'latitude direction',
    'longitude', 'longitude direction']


# In[74]:


geolocations = geolocations.astype({'latitude':'float',
   'longitude':'float'})
geolocations.dtypes


# In[75]:


(geolocations
    .assign(city=cities['City'])
)


# ### How it works...

# In[76]:


geolocations.apply(pd.to_numeric, errors='ignore')


# ### There's more...

# In[77]:


cities.Geolocation.str.split(pat=r'° |, ', expand=True)


# In[78]:


cities.Geolocation.str.extract(r'([0-9.]+). (N|S), ([0-9.]+). (E|W)',
   expand=True)


# ## Tidying when variables are stored in column names and values

# ### Getting ready

# In[79]:


sensors = pd.read_csv('data/sensors.csv')
sensors


# In[80]:


sensors.melt(id_vars=['Group', 'Property'], var_name='Year')


# In[81]:


(sensors
    .melt(id_vars=['Group', 'Property'], var_name='Year') 
    .pivot_table(index=['Group', 'Year'],
                 columns='Property', values='value') 
    .reset_index() 
    .rename_axis(None, axis='columns')
)


# ### How it works...

# ### There's more...

# In[82]:


(sensors
    .set_index(['Group', 'Property']) 
    .stack() 
    .unstack('Property') 
    .rename_axis(['Group', 'Year'], axis='index') 
    .rename_axis(None, axis='columns') 
    .reset_index()
)


# In[ ]:




