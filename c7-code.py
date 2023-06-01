#!/usr/bin/env python
# coding: utf-8

# # Filtering Rows

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

# ## Calculating boolean statistics

# ### How to do it...

# In[2]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
movie[['duration']].head()


# In[3]:


movie_2_hours = movie['duration'] > 120
movie_2_hours.head(10)


# In[4]:


movie_2_hours.sum()


# In[5]:


movie_2_hours.mean()


# In[6]:


movie['duration'].dropna().gt(120).mean()


# In[7]:


movie_2_hours.describe()


# ### How it works...

# In[8]:


movie_2_hours.value_counts(normalize=True)


# In[9]:


movie_2_hours.astype(int).describe()


# ### There's more...

# In[10]:


actors = movie[['actor_1_facebook_likes',
                'actor_2_facebook_likes']].dropna()

(actors['actor_1_facebook_likes'] >
      actors['actor_2_facebook_likes']).mean()


# ## Constructing multiple boolean conditions

# ### How to do it...

# In[11]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')


# In[12]:


criteria1 = movie.imdb_score > 8
criteria2 = movie.content_rating == 'PG-13'
criteria3 = ((movie.title_year < 2000) | (movie.title_year > 2009))


# In[13]:


criteria_final = criteria1 & criteria2 & criteria3
criteria_final.head()


# ### How it works...
 
# ### There's more...

# In[14]:


5 < 10 and 3 > 4


# In[15]:


5 < 10 and 3 > 4


# In[16]:


True and 3 > 4


# In[17]:


True and False


# In[18]:


False


# In[19]:


movie.title_year < 2000 | movie.title_year > 2009


# ## Filtering with boolean arrays

# ### How to do it...

# In[20]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
crit_a1 = movie.imdb_score > 8
crit_a2 = movie.content_rating == 'PG-13'
crit_a3 = (movie.title_year < 2000) | (movie.title_year > 2009)
final_crit_a = crit_a1 & crit_a2 & crit_a3


# In[21]:


crit_b1 = movie.imdb_score < 5
crit_b2 = movie.content_rating == 'R'
crit_b3 = ((movie.title_year >= 2000) &
(movie.title_year <= 2010))
final_crit_b = crit_b1 & crit_b2 & crit_b3


# In[22]:


final_crit_all = final_crit_a | final_crit_b
final_crit_all.head()


# In[23]:


movie[final_crit_all].head()


# In[24]:


movie.loc[final_crit_all].head()
type(final_crit_all) # pandas.core.series.Series

# integer location method에서는 tolist() 또는 to_numpy()로 변환하여
# 리스트나 배열 형태의 불리언 어레이를 만들어 넣어줘야 함
movie.iloc[final_crit_all.tolist(), ].head() 
movie.iloc[final_crit_all.to_numpy(), ].head()
type(final_crit_all.tolist()) # list
type(final_crit_all.to_numpy()) # numpy.ndarray

# 또는 Boolean의 값만 가지고 와서 사용
movie.iloc[final_crit_all.values].head()
type(final_crit_all.values)
# In[25]:


cols = ['imdb_score', 'content_rating', 'title_year']
movie_filtered = movie.loc[final_crit_all, cols]
movie_filtered.head(10)


# ### How it works...

# In[26]:


movie.iloc[final_crit_all]


# In[43]:


movie.iloc[final_crit_all.values]


# ### There's more...

# In[44]:


final_crit_a2 = ((movie.imdb_score > 8) & 
   (movie.content_rating == 'PG-13') & 
   ((movie.title_year < 2000) |
    (movie.title_year > 2009))
)
final_crit_a2.equals(final_crit_a)


# ## Comparing Row Filtering and Index Filtering

# ### How to do it...

# In[45]:


college = pd.read_csv('data/college.csv')
college[college['STABBR'] == 'TX'].head()


# In[46]:


college2 = college.set_index('STABBR')
college2.loc['TX'].head()


# In[47]:


get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")


# In[48]:


get_ipython().run_line_magic('timeit', "college2.loc['TX']")


# In[49]:


get_ipython().run_line_magic('timeit', "college2 = college.set_index('STABBR')")


# ### How it works...

# ### There's more...

# In[50]:


states = ['TX', 'CA', 'NY']
college[college['STABBR'].isin(states)]


# In[51]:


college2.loc[states]


# ## Selecting with unique and sorted indexes

# ### How to do it...

# In[52]:


college = pd.read_csv('data/college.csv')
college2 = college.set_index('STABBR')
college2.index.is_monotonic


# In[53]:


college3 = college2.sort_index()
college3.index.is_monotonic


# In[54]:


get_ipython().run_line_magic('timeit', "college[college['STABBR'] == 'TX']")


# In[55]:


get_ipython().run_line_magic('timeit', "college2.loc['TX']")


# In[56]:


get_ipython().run_line_magic('timeit', "college3.loc['TX']")


# In[57]:


college_unique = college.set_index('INSTNM')
college_unique.index.is_unique


# In[58]:


college[college['INSTNM'] == 'Stanford University']


# In[59]:


college_unique.loc['Stanford University']


# In[60]:


college_unique.loc[['Stanford University']]


# In[61]:


get_ipython().run_line_magic('timeit', "college[college['INSTNM'] == 'Stanford University']")


# In[62]:


get_ipython().run_line_magic('timeit', "college_unique.loc[['Stanford University']]")


# ### How it works...

# ### There's more...

# In[63]:
college.iloc[:, 0:2].iloc[:, [True, False]]

college.index = college['CITY'] + ', ' + college['STABBR']
college = college.sort_index()
college.head()


# In[64]:


college.loc['Miami, FL'].head()


# In[65]:


get_ipython().run_cell_magic('timeit', 
                             '', 
                             "crit1 = college['CITY'] == 'Miami'\ncrit2 = college['STABBR'] == 'FL'\ncollege[crit1 & crit2]")


# In[66]:


get_ipython().run_line_magic('timeit', 
                             "college.loc['Miami, FL']")


# ## Translating SQL WHERE clauses

# ### How to do it...

# In[67]:


employee = pd.read_csv('data/employee.csv')


# In[68]:


employee.dtypes


# In[69]:


employee.DEPARTMENT.value_counts().head()


# In[70]:


employee.GENDER.value_counts()


# In[71]:


employee.BASE_SALARY.describe()


# In[72]:


depts = ['Houston Police Department-HPD',
         'Houston Fire Department (HFD)']

criteria_dept = employee.DEPARTMENT.isin(depts)

criteria_gender = employee.GENDER == 'Female'

criteria_sal = ((employee.BASE_SALARY >= 80000) & (employee.BASE_SALARY <= 120000))


# In[73]:


criteria_final = (criteria_dept &
                  criteria_gender &
                  criteria_sal)
type(criteria_final) # pandas.core.series.Series

criteria_final_test = [criteria_dept &
                       criteria_gender &
                       criteria_sal]
type(criteria_final_test) # list


# In[74]:


select_columns = ['UNIQUE_ID', 'DEPARTMENT', 'GENDER', 'BASE_SALARY']

employee.loc[criteria_final, select_columns].head()


# ### How it works...

# ### There's more...

# In[75]:


criteria_sal = employee.BASE_SALARY.between(80_000, 120_000)


# In[76]:


top_5_depts = employee.DEPARTMENT.value_counts().index[:5]
criteria = ~employee.DEPARTMENT.isin(top_5_depts)
employee[criteria]


# ## Improving readability of boolean indexing with the query method

# ### How to do it...

# In[77]:


employee = pd.read_csv('data/employee.csv')
depts = ['Houston Police Department-HPD',
         'Houston Fire Department (HFD)']
select_columns = ['UNIQUE_ID', 
                  'DEPARTMENT',
                  'GENDER', 
                  'BASE_SALARY']


# In[78]:


qs =( "DEPARTMENT in @depts "
      " and GENDER == 'Female' "
      " and 80000 <= BASE_SALARY <= 120000" )
emp_filtered = employee.query(qs)
emp_filtered[select_columns].head()


# ### How it works...

# ### There's more...

# In[79]:


top10_depts = (employee.DEPARTMENT.value_counts() 
   .index[:10].tolist()
)
qs = "DEPARTMENT not in @top10_depts and GENDER == 'Female'"
employee_filtered2 = employee.query(qs)
employee_filtered2.head()


# ## Preserving Series size with the where method

# ### How to do it...

# In[80]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
fb_likes = movie['actor_1_facebook_likes'].dropna()
fb_likes.head()
type(fb_likes)

# In[81]:


fb_likes.describe()


# In[82]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
fb_likes.hist(ax=ax)
fig.savefig('tmp/c7-hist.png', dpi=300)     # doctest: +SKIP


# In[83]:


criteria_high = fb_likes < 20_000
criteria_high.mean().round(2)


# In[84]:


fb_likes.where(criteria_high).head()


# In[85]:


fb_likes.where(criteria_high, other = 20000).head()



# In[86]:


criteria_low = fb_likes > 300
fb_likes_cap = (fb_likes
   .where(criteria_high, other=20_000)
   .where(criteria_low, 300)
)
fb_likes_cap.head()


# In[87]:


len(fb_likes), len(fb_likes_cap)


# In[88]:


fig, ax = plt.subplots(figsize=(10, 8))
fb_likes_cap.hist(ax=ax)
fig.savefig('tmp/c7-hist2.png', dpi=300)     # doctest: +SKIP


# ### How it works...

# ### There's more...

# In[89]:


fb_likes_cap2 = fb_likes.clip(lower=300, upper=20000)
fb_likes_cap2.equals(fb_likes_cap)


# ## Masking DataFrame rows

# ### How to do it...

# In[90]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['title_year'] >= 2010
c2 = movie['title_year'].isna()
criteria = c1 | c2

movie.title_year

# In[91]:


movie.title_year.mask(criteria).head()


# In[92]:


movie_mask = (movie
    .mask(criteria)
    .dropna(how='all')
)
movie_mask.head()


# In[93]:


movie_boolean = movie[movie['title_year'] < 2010]
movie_mask.equals(movie_boolean)


# In[94]:


movie_mask.shape == movie_boolean.shape


# In[95]:


movie_mask.dtypes == movie_boolean.dtypes
type(movie_mask.dtypes != movie_boolean.dtypes)


movie_mask.loc[:, movie_mask.dtypes != movie_boolean.dtypes]

movie_mask.iloc[:, (movie_mask.dtypes != movie_boolean.dtypes).tolist()]

movie_mask.dtypes[movie_mask.dtypes != movie_boolean.dtypes]
movie_boolean.dtypes[movie_mask.dtypes != movie_boolean.dtypes]


# In[96]:


from pandas.testing import assert_frame_equal
assert_frame_equal(movie_boolean, movie_mask, check_dtype=False)
# 같으면 None 반환

assert_frame_equal(movie_boolean, movie_mask, check_dtype=True)
# 같지 않으므로 오류 발생 
# ### How it works...

# ### There's more...

# In[97]:


get_ipython().run_line_magic('timeit', "movie.mask(criteria).dropna(how='all')")


# In[98]:


get_ipython().run_line_magic('timeit', "movie[movie['title_year'] < 2010]")


# ## Selecting with booleans, integer location, and labels

# ### How to do it...

# In[99]:


movie = pd.read_csv('data/movie.csv', index_col='movie_title')
c1 = movie['content_rating'] == 'G'
c2 = movie['imdb_score'] < 4
criteria = c1 & c2
type(criteria)

# In[100]:


movie_loc = movie.loc[criteria]
movie_loc.head()


# In[101]:


movie_loc.equals(movie[criteria])


# In[102]:


movie_iloc = movie.iloc[criteria]


# In[103]:


movie_iloc = movie.iloc[criteria.values]
movie_iloc.equals(movie_loc)


# In[104]:


criteria_col = movie.dtypes == np.int64

criteria_col.head()
type(criteria_col) # Series


# In[105]:


movie.loc[:, criteria_col].head()


# In[106]:


movie.iloc[:, criteria_col.values].head()


# In[107]:


cols = ['content_rating', 'imdb_score', 'title_year', 'gross']
movie.loc[criteria, cols].sort_values('imdb_score')


# In[108]:


col_index = [movie.columns.get_loc(col) for col in cols]
col_index


# In[109]:


movie.iloc[criteria.values, col_index].sort_values('imdb_score')


# ### How it works...

# In[110]:


a = criteria.values
a[:5]


# In[111]:


len(a), len(criteria)
type(a), type(criteria)

# In[112]:

movie.dtypes.value_counts()

movie.select_dtypes(include = "int64")
movie.select_dtypes(object)
movie.select_dtypes(float)
movie.dtypes
type(movie)

# In[ ]:

movie.columns.isin(movie.columns.sort_values()[0:2])
movi.columnslike


