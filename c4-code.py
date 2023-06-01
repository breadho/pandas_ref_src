#!/usr/bin/env python
# coding: utf-8

# # Beginning Data Analysis 

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
 if not name.startswith('_'):
      del globals()[name]
del(name) # name object 삭제

# In[ ]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ## Introduction

# ## Developing a data analysis routine

# ### How to do it...

# In[ ]:


college = pd.read_csv('data/college.csv')
college.sample(random_state = 42)


# In[ ]:


college.shape


# In[ ]:


college.info()


# In[ ]:


college.describe(include=[np.number]).T

college.shape[0] - college.count()


# In[ ]:

college.describe(include = [np.object]).T
college.describe(include = [pd.Categorical]).T
college.describe(include=[np.object, pd.Categorical]).T


# ### How it works...

# ### There's more...

# In[ ]:


college.describe(include=[np.number],
   percentiles=[.01, .05, .10, .25, .5,
                .75, .9, .95, .99]).T



# ## Data dictionaries

# In[ ]:


pd.read_csv('data/college_data_dictionary.csv')


# ## Reducing memory by changing data types

# ### How to do it...

# In[ ]:
college = pd.read_csv('data/college.csv')
different_cols = ['RELAFFIL', 'SATMTMID', 'CURROPER',
                  'INSTNM', 'STABBR']
col2 = college.loc[:, different_cols]
col2.head()


# In[ ]:


col2.dtypes


# In[ ]:


original_mem = col2.memory_usage(deep=True)
original_mem

col2.info()

# In[ ]:


col2['RELAFFIL'] = col2['RELAFFIL'].astype(np.int8)    


# In[ ]:


col2.dtypes


# In[ ]:


college[different_cols].memory_usage(deep=True)
col2.memory_usage(deep=True)

# In[ ]:


col2.select_dtypes(include = ['object']).nunique()


# In[ ]:


col2['STABBR'] = col2['STABBR'].astype('category')
col2.dtypes


# In[ ]:


new_mem = col2.memory_usage(deep = True)
new_mem


# In[ ]:


new_mem / original_mem


# ### How it works...

# ### There's more...

# In[ ]:


college.loc[0, 'CURROPER'] = 10000000
college.loc[0, 'INSTNM'] = college.loc[0, 'INSTNM'] + 'a'
college[['CURROPER', 'INSTNM']].memory_usage(deep=True)


# In[ ]:


college['MENONLY'].dtype


# In[ ]:


college['MENONLY'].astype(np.int8)


# In[ ]:


college.assign(MENONLY=college['MENONLY'].astype('float16'),
    RELAFFIL=college['RELAFFIL'].astype('int8'))


# In[ ]:


college.index = pd.Int64Index(college.index)
college.index.memory_usage() # previously was just 80


# ## Selecting the smallest of the largest

# ### How to do it...

# In[ ]:

# 이 예제에서는 .nlargest와 .nsmallest 와 같은 편리한 메서드를 사용해 
# 가장 저렴한 예산으로 만들어진 영화 찾기
movie = pd.read_csv('data/movie.csv')
movie2 = movie[['movie_title', 'imdb_score', 'budget']]
movie2.head()


# In[ ]:


movie2.nlargest(100, 'imdb_score').head()


# In[ ]:


(movie2
  .nlargest(100, 'imdb_score')
  .nsmallest(5, 'budget')
)


# ### How it works...

# ### There's more...

# ## Selecting the largest of each group by sorting

# ### How to do it...

# In[ ]:


movie = pd.read_csv('data/movie.csv')
movie[['movie_title', 'title_year', 'imdb_score']]


# In[ ]:

# 내림차순
(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .sort_values('title_year', ascending=False)
)


# In[ ]:


(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .sort_values(['title_year','imdb_score'],
               ascending=False)
)


# In[ ]:


(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .sort_values(['title_year','imdb_score'],
               ascending=False)
  .drop_duplicates(subset='title_year')
)




# ### How it works...

# ## There's more...

# In[ ]:


(movie
  [['movie_title', 'title_year', 'imdb_score']]
  .groupby('title_year', as_index=False)
  .apply(lambda df: 
             df.sort_values('imdb_score',
                            ascending=False)
             .head(1))
  .sort_values('title_year', ascending=False)
)


# In[ ]:


(movie
  [['movie_title', 'title_year', 'content_rating', 'budget']]
   .sort_values(['title_year','content_rating', 'budget'],
                ascending = [False, False, True])
   .drop_duplicates(subset = ['title_year', 'content_rating'])
)


# ## Replicating nlargest with sort_values

# ### How to do it...

# In[ ]:


movie = pd.read_csv('data/movie.csv')

(movie
   [['movie_title', 'imdb_score', 'budget']]
   .nlargest(100, 'imdb_score') 
   .nsmallest(5, 'budget')
)


# In[ ]:


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .sort_values('imdb_score', ascending=False)
   .head(100)
)


# In[ ]:


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .sort_values('imdb_score', ascending=False)
   .head(100) 
   .sort_values('budget')
   .head(5)
)


# ### How it works...

# In[ ]:


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .nlargest(100, 'imdb_score')
   .tail()
)


# In[ ]:


(movie
   [['movie_title', 'imdb_score', 'budget']]
   .sort_values('imdb_score', ascending=False) 
   .head(100)
   .tail()
)


# ## Calculating a trailing stop order price

# ### How to do it...

# In[ ]:

# 설치 명령어
# conda install -c conda-forge requests-cache

import datetime
import pandas_datareader.data as web
import requests_cache

session = (requests_cache.CachedSession(cache_name='cache', 
                                        backend='sqlite', 
                                        expire_after = datetime.timedelta(days=90))
            )

# !pip install requests_cache
# In[ ]:


tsla = web.DataReader('tsla', 
                      data_source = 'yahoo',
                      start = '2017-1-1', 
                      session = session)

# tsla = pd.read_csv("data/TSLA.csv")

tsla.head(8)


# In[ ]:


tsla_close = tsla['Close']


# In[ ]:


tsla_cummax = tsla_close.cummax()
tsla_cummax.head()


# In[ ]:


(tsla
  ['Close']
  .cummax() # 현재까지 종가 중에 최대값 추적
  .mul(.9)
  .head()
)


# ### How it works...

# ### There's more...
