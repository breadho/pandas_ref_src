#!/usr/bin/env python
# coding: utf-8

import os
os.getcwd()
os.chdir('D:/pandas_cookbook')

# # Chapter 2: Essential DataFrame Operations

# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# In[4]:


dir()


# ## Introduction

# ## Selecting Multiple DataFrame Columns

# ### How to do it\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
movie_actor_director = movies[['actor_1_name', 
                               'actor_2_name',
                               'actor_3_name', 
                               'director_name']]
movie_actor_director.head()

# movie_actor_director.shape

# In[ ]:


type(movies[['director_name']])


# In[ ]:


type(movies['director_name'])


# In[ ]:


type(movies.loc[:, ['director_name']])


# In[ ]:


t3ype(movies.loc[:, 'director_name'])


# ### How it works\...

# ### There\'s more\...

# In[ ]:


cols = ['actor_1_name', 
        'actor_2_name',
        'actor_3_name', 
        'director_name']

movie_actor_director = movies[cols]


# In[ ]:


# key error 발생 사례

movies['actor_1_name', 
       'actor_2_name',
       'actor_3_name', 
       'director_name']


# ## Selecting Columns with Methods

# ### How it works\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
def shorten(col):
    return (col.replace('facebook_likes', 'fb')
               .replace('_for_reviews', '')
    )

movies = movies.rename(columns=shorten)

movies.dtypes.value_counts()


# In[ ]:


movies.select_dtypes(include='int').head()

movies.select_dtypes(include='int64').head()


# In[ ]:


movies.select_dtypes(include='number').head()

#int랑 float이 모두 추출됨


# In[ ]:


movies.select_dtypes(include=['int64', 'object']).head()


# In[ ]:


movies.select_dtypes(exclude='float').head()


# In[ ]:


movies.filter(like='fb').head()


# In[ ]:


cols = ['actor_1_name', 'actor_2_name','actor_3_name', 'director_name']

movies.filter(items=cols).head()


# In[ ]:


movies.filter(regex = r'\d').head()


# ### How it works\...

# ### There\'s more\...

# ### See also

# ## Ordering Column Names

# ### How to do it\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
def shorten(col):
    return (col.replace('facebook_likes', 'fb')
               .replace('_for_reviews', '')
    )
movies = movies.rename(columns = shorten)


# In[ ]:


movies.columns


# In[ ]:


cat_core = ['movie_title', 
            'title_year',
            'content_rating', 
            'genres']

cat_people = ['director_name', 
              'actor_1_name',
              'actor_2_name', 
              'actor_3_name']

cat_other = ['color', 
             'country', 
             'language',
             'plot_keywords', 
             'movie_imdb_link']

cont_fb = ['director_fb', 
           'actor_1_fb',
           'actor_2_fb', 
           'actor_3_fb',
           'cast_total_fb', 
           'movie_fb']

cont_finance = ['budget', 'gross']

cont_num_reviews = ['num_voted_users', 
                    'num_user',
                    'num_critic']

cont_other = ['imdb_score', 
              'duration',
               'aspect_ratio', 
               'facenumber_in_poster']


# In[ ]:


new_col_order = cat_core + cat_people + \
                cat_other + \
                cont_fb + \
                cont_finance + \
                cont_num_reviews + \
                cont_other

print(new_col_order)
                
set(movies.columns) == set(new_col_order)


# In[ ]:


movies[new_col_order].head()

type(movies[new_col_order])
type(new_col_order)


# ### How it works\...

# ### There\'s more\...

# ### See also

# ## Summarizing a DataFrame

# ### How to do it\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
movies.shape


# In[ ]:


movies.size


# In[ ]:


movies.ndim


# In[ ]:


len(movies)


# In[ ]:


movies.count()


# In[ ]:


movies.min()


# In[ ]:


movies.describe().T
type(movies.describe().T)

# print(movies.describe().T)
# In[ ]:


movies.describe(percentiles=[.01, .3, .99]).T


# ### How it works\...

# ### There\'s more\...

# In[ ]:


movies.min(skipna = False)


# ## Chaining DataFrame Methods

# ### How to do it\...

# In[ ]:


movies = pd.read_csv('data/movie.csv')
def shorten(col):
    return (col.replace('facebook_likes', 'fb')
               .replace('_for_reviews', '')
    )
movies = movies.rename(columns=shorten)
movies.isnull().head()


# In[ ]:


(movies
   .isnull()
   .sum()
  # .head()
)

# In[ ]:

# 총 결측치 개수 확인
movies.isnull().sum().sum()


# In[ ]:


movies.isnull().any().any()

# movies.isnull().any().all()

# ### How it works\...

# In[ ]:


movies.isnull().dtypes.value_counts()


# ### There\'s more\...

# In[ ]:

# 기본 집계 메서드는 결측치가 있을 때, 아무것도 반환하지 않음
movies[['color', 'movie_title', 'color']].max()


# In[ ]:

# 각 열에 대해 무언가를 반환하게 하려면 결측치를 채워야 함
with pd.option_context('max_colwidth', 20):
    movies.select_dtypes(['object']).fillna('').max()


# In[ ]:


with pd.option_context('max_colwidth', 20):
    (movies
        .select_dtypes(['object'])
        .fillna('')
        .max()
    )

# ### See also

# ## DataFrame Operations

# In[ ]:


colleges = pd.read_csv('data/college.csv')
colleges + 5


# In[ ]:


colleges = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = colleges.filter(like='UGDS_')
college_ugds.head()

college_ugds.index

# In[ ]:


name = 'Northwest-Shoals Community College'
college_ugds.loc[name]


# In[ ]:

# bankers rounding -> college_ugds.loc[name].round(2)


# In[ ]:


(college_ugds.loc[name] + .0001).round(2)


# In[ ]:


college_ugds + .00501


# In[ ]:


(college_ugds + .00501) // .01


# In[ ]:


college_ugds_op_round = (college_ugds + .00501) // .01 / 100
college_ugds_op_round.head()


# In[ ]:


college_ugds_round = (college_ugds + .00001).round(2)
college_ugds_round


# In[ ]:


college_ugds_op_round.equals(college_ugds_round)


# ### How it works\...

# In[ ]:

# 부동소수점
.045 + .005


# ### There\'s more\...

# In[ ]:


college2 = (college_ugds
    .add(.00501) 
    .floordiv(.01) 
    .div(100)
)
college2.equals(college_ugds_op_round)


# ### See also

# ## Comparing Missing Values

# In[ ]:

# 결측치 비교
np.nan == np.nan


# In[ ]:


None == None


# In[ ]:


np.nan > 5


# In[ ]:


5 > np.nan


# In[ ]:


np.nan != 5


# ### Getting ready

# In[ ]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')


# In[ ]:


college_ugds == .0019


# In[ ]:


college_self_compare = college_ugds == college_ugds
college_self_compare.head()


# In[ ]:


college_self_compare.all()


# In[ ]:


(college_ugds == np.nan).sum()


# In[ ]:


college_ugds.isnull().sum()


# In[ ]:


college_ugds.equals(college_ugds)


# ### How it works\...

# ### There\'s more\...

# In[ ]:


college_ugds.eq(.0019)    # same as college_ugds == .0019


# In[ ]:


from pandas.testing import assert_frame_equal
assert_frame_equal(college_ugds, college_ugds) is None


# ## Transposing the direction of a DataFrame operation

# ### How to do it\...

# In[ ]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')
college_ugds.head()


# In[ ]:

college.shape
college_ugds.count()


# In[ ]:


college_ugds.count(axis='columns').head()


# In[ ]:


college_ugds.sum(axis='columns').head()


# In[ ]:


college_ugds.median(axis='index')


# ### How it works\...

# ### There\'s more\...

# In[ ]:


college_ugds_cumsum = college_ugds.cumsum(axis=1)
college_ugds_cumsum.head()

# ### See also

# ## Determining college campus diversity

# In[ ]:


pd.read_csv('data/college_diversity.csv', index_col='School')


# ### How to do it\...

# In[ ]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')


# In[ ]:


(college_ugds.isnull()
   .sum(axis='columns')
   .sort_values(ascending=False)
   .head()
)


# In[ ]:


college_ugds = college_ugds.dropna(how='all')
college_ugds.isnull().sum()


# In[ ]:


college_ugds.ge(.15)


# In[ ]:


diversity_metric = college_ugds.ge(.15).sum(axis='columns')
diversity_metric.head()


# In[ ]:


diversity_metric.value_counts()


# In[ ]:


diversity_metric.sort_values(ascending=False).head()


# In[ ]:


college_ugds.loc[['Regency Beauty Institute-Austin',
                   'Central Texas Beauty College-Temple']]


# In[ ]:


us_news_top = ['Rutgers University-Newark',
                  'Andrews University',
                  'Stanford University',
                  'University of Houston',
                  'University of Nevada-Las Vegas']
diversity_metric.loc[us_news_top]


# ### How it works\...

# ### There\'s more\...

# In[ ]:


(college_ugds
   .max(axis=1)
   .sort_values(ascending=False)
   .head(10)
)


# In[ ]:


(college_ugds > .01).all(axis=1).any()



# ### See also
