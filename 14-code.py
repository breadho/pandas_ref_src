#!/usr/bin/env python
# coding: utf-8

# # Debugging and Testing Pandas

# ## Code to Transform Data

# ### How to do it...

# In[1]:


import pandas as pd
import numpy as np
import zipfile
url = 'data/kaggle-survey-2018.zip'


# In[2]:


with zipfile.ZipFile(url) as z:
    print(z.namelist())
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    df = kag.iloc[1:]


# In[3]:


df.T


# In[4]:


df.dtypes


# In[5]:


df.Q1.value_counts(dropna=False)


# In[6]:


def tweak_kag(df):
    na_mask = df.Q9.isna()
    hide_mask = df.Q9.str.startswith('I do not').fillna(False)
    df = df[~na_mask & ~hide_mask]


    q1 = (df.Q1
      .replace({'Prefer not to say': 'Another',
               'Prefer to self-describe': 'Another'})
      .rename('Gender')
    )
    q2 = df.Q2.str.slice(0,2).astype(int).rename('Age')
    def limit_countries(val):
        if val in  {'United States of America', 'India', 'China'}:
            return val
        return 'Another'
    q3 = df.Q3.apply(limit_countries).rename('Country')


    q4 = (df.Q4
     .replace({'Master’s degree': 18,
     'Bachelor’s degree': 16,
     'Doctoral degree': 20,
     'Some college/university study without earning a bachelor’s degree': 13,
     'Professional degree': 19,
     'I prefer not to answer': None,
     'No formal education past high school': 12})
     .fillna(11)
     .rename('Edu')
    )


    def only_cs_stat_val(val):
        if val not in {'cs', 'eng', 'stat'}:
            return 'another'
        return val


    q5 = (df.Q5
            .replace({
                'Computer science (software engineering, etc.)': 'cs',
                'Engineering (non-computer focused)': 'eng',
                'Mathematics or statistics': 'stat'})
             .apply(only_cs_stat_val)
             .rename('Studies'))
    def limit_occupation(val):
        if val in {'Student', 'Data Scientist', 'Software Engineer', 'Not employed',
                  'Data Engineer'}:
            return val
        return 'Another'


    q6 = df.Q6.apply(limit_occupation).rename('Occupation')


    q8 = (df.Q8
      .str.replace('+', '')
      .str.split('-', expand=True)
      .iloc[:,0]
      .fillna(-1)
      .astype(int)
      .rename('Experience')
    )


    q9 = (df.Q9
     .str.replace('+','')
     .str.replace(',','')
     .str.replace('500000', '500')
     .str.replace('I do not wish to disclose my approximate yearly compensation','')
     .str.split('-', expand=True)
     .iloc[:,0]
     .astype(int)
     .mul(1000)
     .rename('Salary'))
    return pd.concat([q1, q2, q3, q4, q5, q6, q8, q9], axis=1)


# In[7]:


tweak_kag(df)


# In[8]:


tweak_kag(df).dtypes


# ### How it works...

# In[9]:


kag = tweak_kag(df)
(kag
    .groupby('Country')
    .apply(lambda g: g.Salary.corr(g.Experience))
)


# ## Apply Performance

# ### How to do it...

# In[13]:


def limit_countries(val):
     if val in  {'United States of America', 'India', 'China'}:
         return val
     return 'Another'


# In[14]:


get_ipython().run_cell_magic('timeit', '', "q3 = df.Q3.apply(limit_countries).rename('Country')")


# In[15]:


get_ipython().run_cell_magic('timeit', '', "other_values = df.Q3.value_counts().iloc[3:].index\nq3_2 = df.Q3.replace(other_values, 'Another')")


# In[16]:


get_ipython().run_cell_magic('timeit', '', "values = {'United States of America', 'India', 'China'}\nq3_3 = df.Q3.where(df.Q3.isin(values), 'Another')")


# In[17]:


get_ipython().run_cell_magic('timeit', '', "values = {'United States of America', 'India', 'China'}\nq3_4 = pd.Series(np.where(df.Q3.isin(values), df.Q3, 'Another'), \n     index=df.index)")


# In[18]:


q3.equals(q3_2)


# In[ ]:


q3.equals(q3_3)


# In[ ]:


q3.equals(q3_4)


# ### How it works...

# ### There's more...

# In[19]:


def limit_countries(val):
     if val in  {'United States of America', 'India', 'China'}:
         return val
     return 'Another'


# In[20]:


q3 = df.Q3.apply(limit_countries).rename('Country')


# In[21]:


def debug(something):
    # what is something? A cell, series, dataframe?
    print(type(something), something)
    1/0


# In[22]:


q3.apply(debug)


# In[28]:


the_item = None
def debug(something):
    global the_item
    the_item = something
    return something


# In[29]:


_ = q3.apply(debug)


# In[30]:


the_item


# ## Improving Apply Performance with Dask, Pandarell, Swifter, and More

# ### How to do it...

# In[31]:


from pandarallel import pandarallel
pandarallel.initialize()


# In[32]:


def limit_countries(val):
     if val in  {'United States of America', 'India', 'China'}:
         return val
     return 'Another'


# In[33]:


get_ipython().run_cell_magic('timeit', '', "res_p = df.Q3.parallel_apply(limit_countries).rename('Country')")


# In[41]:


import swifter


# In[42]:


get_ipython().run_cell_magic('timeit', '', "res_s = df.Q3.swifter.apply(limit_countries).rename('Country')")


# In[43]:


import dask


# In[44]:


get_ipython().run_cell_magic('timeit', '', "res_d = (dask.dataframe.from_pandas(\n       df, npartitions=4)\n   .map_partitions(lambda df: df.Q3.apply(limit_countries))\n   .rename('Countries')\n)")


# In[45]:


np_fn = np.vectorize(limit_countries)


# In[39]:


get_ipython().run_cell_magic('timeit', '', "res_v = df.Q3.apply(np_fn).rename('Country')")


# In[46]:


from numba import jit


# In[50]:


@jit
def limit_countries2(val):
     if val in  ['United States of America', 'India', 'China']:
         return val
     return 'Another'


# In[51]:


get_ipython().run_cell_magic('timeit', '', "res_n = df.Q3.apply(limit_countries2).rename('Country')")


# ### How it works...

# ## Inspecting Code 

# ### How to do it...

# In[52]:


import zipfile
url = 'data/kaggle-survey-2018.zip'


# In[53]:


with zipfile.ZipFile(url) as z:
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    df = kag.iloc[1:]


# In[54]:


get_ipython().run_line_magic('pinfo', 'df.Q3.apply')


# In[55]:


get_ipython().run_line_magic('pinfo2', 'df.Q3.apply')


# In[56]:


import pandas.core.series
pandas.core.series.lib


# In[57]:


get_ipython().run_line_magic('pinfo2', 'pandas.core.series.lib.map_infer')


# ### How it works...

# ### There's more...

# ## Debugging in Jupyter

# ### How to do it...

# In[58]:


import zipfile
url = 'data/kaggle-survey-2018.zip'


# In[59]:


with zipfile.ZipFile(url) as z:
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    df = kag.iloc[1:]


# In[60]:


def add1(x):
    return x + 1


# In[61]:


df.Q3.apply(add1)


# In[62]:


from IPython.core.debugger import set_trace


# In[63]:


def add1(x):
    set_trace()
    return x + 1


# In[ ]:


df.Q3.apply(add1)


# ### How it works...

# ### There's more...

# ##  Managing data integrity with Great Expectations

# ### How to do it...

# In[64]:


kag = tweak_kag(df)


# In[66]:


import great_expectations as ge
kag_ge = ge.from_pandas(kag)


# In[67]:


sorted([x for x in set(dir(kag_ge)) - set(dir(kag))
    if not x.startswith('_')])


# In[68]:


kag_ge.expect_column_to_exist('Salary')


# In[69]:


kag_ge.expect_column_mean_to_be_between(
   'Salary', min_value=10_000, max_value=100_000)


# In[70]:


kag_ge.expect_column_values_to_be_between(
   'Salary', min_value=0, max_value=500_000)


# In[71]:


kag_ge.expect_column_values_to_not_be_null('Salary')


# In[72]:


kag_ge.expect_column_values_to_match_regex(
    'Country', r'America|India|Another|China')


# In[73]:


kag_ge.expect_column_values_to_be_of_type(
   'Salary', type_='int')


# In[74]:


kag_ge.save_expectation_suite('kaggle_expectations.json')


# In[75]:


kag_ge.to_csv('kag.csv')
import json
ge.validate(ge.read_csv('kag.csv'), 
    expectation_suite=json.load(
        open('kaggle_expectations.json')))


# ### How it works...

# ## Using pytest with pandas

# ### How to do it...

# ### How it works...

# ### There's more...

# ## Generating Tests with Hypothesis

# ### How to do it...

# ### How it works...
