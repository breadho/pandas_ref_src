#!/usr/bin/env python
# coding: utf-8

# # Chapter 1: Pandas Foundations

# In[1]:


import pandas as pd
import numpy as np


# ## Introduction

# ## Dissecting the anatomy of a DataFrame

# In[2]:


pd.set_option('max_columns', 4, 'max_rows', 10)

import session_info
session_info.show()


# In[3]:


movies = pd.read_csv('data/movie.csv')
movies.head()


# ### How it works...

# ## DataFrame Attributes

# ### How to do it... {#how-to-do-it-1}

# In[4]:


movies = pd.read_csv('data/movie.csv')
columns = movies.columns
index = movies.index
data = movies.values


# In[5]:


columns


# In[6]:


index


# In[7]:


data


# In[8]:


type(index)


# In[9]:


type(columns)


# In[10]:


type(data)


# In[11]:


issubclass(pd.RangeIndex, pd.Index)


# ### How it works...

# ### There's more

# In[12]:


index.values


# In[13]:


columns.values


# ## Understanding data types

# ### How to do it... {#how-to-do-it-2}

# In[14]:


movies = pd.read_csv('data/movie.csv')


# In[15]:


movies.dtypes


# In[16]:


movies.dtypes.value_counts()


# In[17]:


movies.info()


# ### How it works...

# In[18]:


pd.Series(['Paul', np.nan, 'George']).dtype


# ### There's more...

# ### See also

# ## Selecting a Column

# ### How to do it... {#how-to-do-it-3}

# In[21]:


movies = pd.read_csv('data/movie.csv')
movies['director_name']


# In[22]:


movies.director_name


# In[23]:


movies.loc[:, 'director_name']


# In[25]:


movies.iloc[:, 1]


# In[26]:


movies['director_name'].index


# In[35]:


mdf = movies.copy()


# In[36]:


mdf.index


# In[27]:


movies['director_name'].dtype


# In[38]:


mdf.dtypes


# In[28]:


movies['director_name'].size


# In[39]:


mdf.size


# In[48]:


mdf.ndim


# In[50]:


movies['director_name'].shape


# In[51]:


mdf.shape


# In[52]:


movies['director_name'].name


# In[54]:


mdf.name


# In[45]:


mdf.index


# In[30]:


type(movies['director_name'])


# In[57]:


type(movies['director_name'])


# In[58]:


type(mdf)


# In[66]:


mdf.dtypes


# In[59]:


movies['director_name'].apply(type).


# In[65]:


movies['director_name'].apply(type).value_counts()


# In[67]:


movies['director_name'].apply(type)


# In[ ]:





# ### How it works...

# ### There's more

# ### See also

# ## Calling Series Methods

# In[68]:


s_attr_methods = set(dir(pd.Series))
len(s_attr_methods)


# In[69]:


df_attr_methods = set(dir(pd.DataFrame))
len(df_attr_methods)


# In[70]:


len(s_attr_methods & df_attr_methods)


# ### How to do it... {#how-to-do-it-4}

# In[71]:


movies = pd.read_csv('data/movie.csv')
director = movies['director_name']
fb_likes = movies['actor_1_facebook_likes']


# In[72]:


director.dtype


# In[73]:


fb_likes.dtype


# In[74]:


director.head()


# In[76]:


mdf.head()


# In[77]:


director.sample(n=5, random_state=42)


# In[78]:


mdf.sample(n = 5, random_state = 42)


# In[89]:


fb_likes.head(10)


# In[80]:


director.value_counts()


# In[82]:


fb_likes.value_counts()


# In[83]:


director.size


# In[84]:


director.shape


# In[85]:


len(director)


# In[86]:


len(mdf)


# In[87]:


director.unique()


# In[88]:


director.count()


# In[90]:


mdf.count()


# In[91]:


len(mdf) - mdf.count()


# In[92]:


len(director) - director.count()


# In[93]:


fb_likes.count()


# In[94]:


fb_likes.quantile()


# In[95]:


fb_likes.min()


# In[96]:


fb_likes.max()


# In[97]:


fb_likes.mean()


# In[98]:


fb_likes.median()


# In[99]:


fb_likes.std()


# In[100]:


fb_likes.describe()


# In[101]:


director.describe()


# In[104]:


mdf.boxplot()


# In[105]:


fb_likes.quantile(.2)


# In[106]:


fb_likes.quantile([.1, .2, .3, .4, .5, .6, .7, .8, .9])


# In[115]:


fb_likes.quantile(np.linspace(start=0, stop=1, num=11))


# In[114]:


np.linspace(start=0, stop=1, num=11)


# In[116]:


director.isna()


# In[117]:


director.isna().value_counts()


# In[119]:


mdf.isna()


# In[120]:


fb_likes_filled = fb_likes.fillna(0)
fb_likes_filled.count()


# In[121]:


fb_likes_dropped = fb_likes.dropna()
fb_likes_dropped.size


# In[129]:


np.array(mdf.shape) - np.array(mdf.dropna().shape)


# In[144]:


mdfdes = mdf.describe()
mdfdes.iloc[0:1,:]


# In[145]:


type(mdfdes.iloc[0:1, :])


# In[147]:


mdfdes.iloc[0:1, :].transpose()


# In[151]:


type(mdfdes.iloc[0:1, :].transpose())


# ### How it works...

# ### There's more...

# In[154]:


director.value_counts(normalize=True)


# In[155]:


director.hasnans


# In[156]:


director.notna()


# In[157]:


director.notna().value_counts()`


# In[158]:


director.isnull()


# In[159]:


director.isnull().value_counts()


# ### See also

# ## Series Operations

# In[ ]:


5 + 9    # plus operator example. Adds 5 and 9


# ### How to do it... {#how-to-do-it-5}

# In[160]:


movies = pd.read_csv('data/movie.csv')
imdb_score = movies['imdb_score']
imdb_score


# In[ ]:


imdb_score + 1


# In[161]:


imdb_score * 2.5


# In[162]:


imdb_score // 7


# In[163]:


imdb_score > 7


# In[164]:


director = movies['director_name']
director == 'James Cameron'


# ### How it works...

# ### There's more...

# In[ ]:


imdb_score.add(1)   # imdb_score + 1


# In[ ]:


imdb_score.gt(7)   # imdb_score > 7


# ### See also

# ## Chaining Series Methods

# ### How to do it... {#how-to-do-it-6}

# In[165]:


movies = pd.read_csv('data/movie.csv')
fb_likes = movies['actor_1_facebook_likes']
director = movies['director_name']


# In[166]:


director.value_counts().head(3)


# In[168]:


fb_likes.isna().value_counts()


# In[167]:


fb_likes.isna().sum() #True가 1


# In[169]:


fb_likes.dtype


# In[170]:


(fb_likes.fillna(0)
         .astype(int)
         .head()
)


# ### How it works...

# ### There's more...

# In[171]:


(fb_likes.fillna(0)
         #.astype(int)
         #.head()
)


# In[172]:


(fb_likes.fillna(0)
         .astype(int)
         #.head()
)


# In[173]:


fb_likes.isna().mean()


# In[178]:


fb_likes.fillna(0)         # .astype(int) \
        # .head()


# In[179]:


def debug_df(df):
    print("BEFORE")
    print(df)
    print("AFTER")
    return df


# In[180]:


(fb_likes.fillna(0)
         .pipe(debug_df)
         .astype(int) 
         .head()
)


# In[181]:


intermediate = None
def get_intermediate(df):
    global intermediate
    intermediate = df
    return df


# In[182]:


res = (fb_likes.fillna(0)
         .pipe(get_intermediate)
         .astype(int) 
         .head()
)


# In[183]:


intermediate


# ## Renaming Column Names

# ### How to do it...

# In[199]:


movies = pd.read_csv('data/movie.csv')


# In[200]:


col_map = {'director_name':'Director Name', 
             'num_critic_for_reviews': 'Critical Reviews'} 


# In[201]:


movies.rename(columns=col_map).head()


# ### How it works... {#how-it-works-8}

# ### There's more {#theres-more-7}

# In[202]:


idx_map = {'Avatar':'Ratava', 'Spectre': 'Ertceps',
  "Pirates of the Caribbean: At World's End": 'POC'}
col_map = {'aspect_ratio': 'aspect',
  "movie_facebook_likes": 'fblikes'}
(movies
   .set_index('movie_title')
   .rename(index=idx_map, columns=col_map)
   .head(4)
)


# In[203]:


movies.set_index('movie_title').index.tolist()


# In[204]:


movies = pd.read_csv('data/movie.csv', index_col='movie_title')
ids = movies.index.tolist()
columns = movies.columns.tolist()


# In[208]:


ids[0:6]


# # rename the row and column labels with list assignments

# In[209]:


ids[0] = 'Ratava'
ids[1] = 'POC'
ids[2] = 'Ertceps'
columns[1] = 'director'
columns[-2] = 'aspect'
columns[-1] = 'fblikes'
movies.index = ids
movies.columns = columns


# In[211]:


movies.head(5)


# In[212]:


def to_clean(val):
    return val.strip().lower().replace(' ', '_')


# In[213]:


movies.rename(columns=to_clean).head(3)


# In[214]:


cols = [col.strip().lower().replace(' ', '_')
        for col in movies.columns]
movies.columns = cols
movies.head(3)


# ## Creating and Deleting columns

# ### How to do it... {#how-to-do-it-9}

# In[219]:


movies = pd.read_csv('data/movie.csv')
movies['has_seen'] = 1


# In[220]:


idx_map = {'Avatar':'Ratava', 'Spectre': 'Ertceps',
  "Pirates of the Caribbean: At World's End": 'POC'}
col_map = {'aspect_ratio': 'aspect',
  "movie_facebook_likes": 'fblikes'}
(movies
   .rename(index=idx_map, columns=col_map)
   .assign(has_seen=0)
)


# In[221]:


movies.head()


# In[222]:


total = (movies['actor_1_facebook_likes'] +
         movies['actor_2_facebook_likes'] + 
         movies['actor_3_facebook_likes'] + 
         movies['director_facebook_likes'])


# In[223]:


total.head(5)


# In[224]:


cols = ['actor_1_facebook_likes','actor_2_facebook_likes',
    'actor_3_facebook_likes','director_facebook_likes']
sum_col = movies[cols].sum(axis='columns')
sum_col.head(5)


# In[225]:


movies.assign(total_likes=sum_col).head(5)


# In[226]:


movies.head(5)


# In[227]:


movies.assign(total_likes = movies[cols].sum(axis='columns')).head(5)


# In[228]:


def sum_likes(df):
   return df[[c for c in df.columns
              if 'like' in c]].sum(axis=1)


# In[229]:


movies.assign(total_likes=sum_likes).head(5)


# In[230]:


(movies
   .assign(total_likes=sum_col)
   ['total_likes']
   .isna()
   .sum()
)


# In[231]:


(movies
   .assign(total_likes=total)
   ['total_likes']
   .isna()
   .sum()
)


# In[232]:


(movies
   .assign(total_likes=total.fillna(0))
   ['total_likes']
   .isna()
   .sum()
)


# In[233]:


def cast_like_gt_actor_director(df):
    return df['cast_total_facebook_likes'] >=            df['total_likes']


# In[234]:


df2 = (movies
   .assign(total_likes=total,
           is_cast_likes_more = cast_like_gt_actor_director)
)


# In[235]:


df2['is_cast_likes_more'].all()


# In[236]:


df2 = df2.drop(columns='total_likes')


# In[238]:


actor_sum = (movies
   [[c for c in movies.columns if 'actor_' in c and '_likes' in c]]
   .sum(axis='columns')
)


# In[239]:


actor_sum.head(5)


# In[240]:


movies['cast_total_facebook_likes'] >= actor_sum


# In[241]:


movies['cast_total_facebook_likes'].ge(actor_sum)


# In[242]:


movies['cast_total_facebook_likes'].ge(actor_sum).all()


# In[252]:


pct_like = (actor_sum
    .div(movies['cast_total_facebook_likes'])
).mul(100)


# In[253]:


pct_like.describe()


# In[254]:


type(pct_like)


# In[261]:


pct_like.values


# In[256]:


pd.Series(pct_like.values,
    index=movies['movie_title'].values).head()


# ### How it works... {#how-it-works-9}

# ### There's more... {#theres-more-8}

# In[264]:


profit_index = movies.columns.get_loc('gross') + 1
profit_index


# In[265]:


movies.insert(loc=profit_index,
              column='profit',
              value=movies['gross'] - movies['budget'])


# In[266]:


del movies['director_name']


# ### See also
