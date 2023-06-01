#!/usr/bin/env python
# coding: utf-8

# # Index Alignment
import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
    if not name.startswith('_'):
        del globals()[name]
        
del(name)

# In[3]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 50)


# ## Introduction

# ## Examining the Index object

# ### How to do it...

# In[12]:


college = pd.read_csv('data/college.csv')
columns = college.columns
columns


# In[13]:


columns.values

type(columns) # pandas.core.indexes.base.Index

# In[14]:


columns[5]


# In[15]:


columns[[1,8,10]]


# In[16]:


columns[-7:-4]


# In[17]:


columns.min(), columns.max(), columns.isnull().sum()


# In[18]:


columns + '_A'


# In[19]:


columns > 'G'


# In[20]:


columns[1] = 'city'


# ### How it works...

# ### There's more...

# In[61]:


c1 = columns[:4]
c1


# In[62]:


c2 = columns[2:6]
c2


# In[63]:


c1.union(c2) # or `c1 | c2`
c1|c2

# In[64]:

# 합집합 - 교집합 
c1.symmetric_difference(c2) # or `c1 ^ c2`


# ## Producing Cartesian products

# ### How to do it...

# In[65]:


s1 = pd.Series(index=list('aaab'), data=np.arange(4))
s1


# In[66]:


s2 = pd.Series(index=list('cababb'), data=np.arange(6))
s2


# In[67]:


s1 + s2


# ### How it works...

# ### There's more...

# In[68]:


s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('aaabb'), data=np.arange(5))
s1 + s2


# In[69]:


s1 = pd.Series(index=list('aaabb'), data=np.arange(5))
s2 = pd.Series(index=list('bbaaa'), data=np.arange(5))
s1 + s2


# In[70]:


s3 = pd.Series(index=list('ab'), data=np.arange(2))
s4 = pd.Series(index=list('ba'), data=np.arange(2))
s3 + s4


# ## Exploding indexes

# ### How to do it...

# In[71]:


employee = pd.read_csv('data/employee.csv', index_col='RACE')
employee.head()


# In[72]:


salary1 = employee['BASE_SALARY']
salary2 = employee['BASE_SALARY']
salary1 is salary2


# In[73]:


salary2 = employee['BASE_SALARY'].copy()
salary1 is salary2

employee['BASE_SALARY'] is salary1


# In[74]:


salary1 = salary1.sort_index()
salary1.head()


# In[75]:


salary2.head()


# In[76]:


salary_add = salary1 + salary2


# In[77]:


salary_add.head()


# In[78]:


salary_add1 = salary1 + salary1
len(salary1), len(salary2), len(salary_add), len(salary_add1)


# ### How it works...

# ### There's more...

# In[79]:


index_vc = salary1.index.value_counts(dropna=False)
(index_vc**2).sum()


# In[80]:


index_vc.pow(2).sum()


# ## Filling values with unequal indexes

# In[4]:


baseball_14 = pd.read_csv('data/baseball14.csv',
   index_col='playerID')
baseball_15 = pd.read_csv('data/baseball15.csv',
   index_col='playerID')
baseball_16 = pd.read_csv('data/baseball16.csv',
   index_col='playerID')
baseball_14.head()


# In[82]:


baseball_14.index.difference(baseball_15.index)


# In[83]:


baseball_14.index.difference(baseball_16.index)


# In[84]:


hits_14 = baseball_14['H']
hits_15 = baseball_15['H']
hits_16 = baseball_16['H']
hits_14.head()


# In[85]:


(hits_14 + hits_15).head()


# In[86]:

# 결측치는 0으로 채우고 합산
hits_14.add(hits_15, fill_value=0).head()


# In[87]:


hits_total = (hits_14
   .add(hits_15, fill_value=0)
   .add(hits_16, fill_value=0)
)
hits_total.head()


# In[88]:


hits_total.hasnans


# ### How it works...

# In[89]:


s = pd.Series(index=['a', 'b', 'c', 'd'],
              data=[np.nan, 3, np.nan, 1])
s


# In[90]:


s1 = pd.Series(index=['a', 'b', 'c'], data=[np.nan, 6, 10])
s1


# In[91]:


s.add(s1, fill_value=5)


# ### There's more...

# In[5]:


df_14 = baseball_14[['G','AB', 'R', 'H']]
df_14.head()


# In[6]:


df_15 = baseball_15[['AB', 'R', 'H', 'HR']]
df_15.head()


# In[7]:


(df_14 + df_15).head(10).style.highlight_null('yellow')


# In[8]:


(df_14
.add(df_15, fill_value=0)
.head(10)
.style.highlight_null('yellow')
)


# ## Adding columns from different DataFrames

# ### How to do it...

# In[94]:


employee = pd.read_csv('data/employee.csv')
dept_sal = employee[['DEPARTMENT', 'BASE_SALARY']]


# In[95]:


dept_sal = dept_sal.sort_values(['DEPARTMENT', 'BASE_SALARY'],
    ascending=[True, False])


# In[96]:

# 부서별 최대값만 남기고 제거함 
max_dept_sal = dept_sal.drop_duplicates(subset='DEPARTMENT')
max_dept_sal.head()


# In[97]:

# 부서 행을 인덱스로 일치시켜 줌
max_dept_sal = max_dept_sal.set_index('DEPARTMENT')
employee = employee.set_index('DEPARTMENT')


# In[98]:

employee = (employee
   .assign(MAX_DEPT_SALARY = max_dept_sal['BASE_SALARY'])
)

employee
employee.shape
len(employee.index)


# In[99]:


employee.query('BASE_SALARY > MAX_DEPT_SALARY')


# In[100]:


employee = pd.read_csv('data/employee.csv')
max_dept_sal = (employee
    [['DEPARTMENT', 'BASE_SALARY']]
    .sort_values(['DEPARTMENT', 'BASE_SALARY'],
        ascending=[True, False])
    .drop_duplicates(subset = 'DEPARTMENT')
    .set_index('DEPARTMENT')
)


# In[101]:


(employee
   .set_index('DEPARTMENT')
   .assign(MAX_DEPT_SALARY=max_dept_sal['BASE_SALARY'])
)


# ### How it works...

# In[102]:


random_salary = (dept_sal
    .sample(n=10, random_state=42)
    .set_index('DEPARTMENT')
)
random_salary


# In[103]:


employee['RANDOM_SALARY'] = random_salary['BASE_SALARY']


# ### There's more...

# In[104]:


(employee
    .set_index('DEPARTMENT')
    .assign(MAX_SALARY2=max_dept_sal['BASE_SALARY'].head(3))
    .MAX_SALARY2.index
    # .value_counts()
)

max_dept_sal

# In[105]:


max_sal = (employee
    .groupby('DEPARTMENT')
    .BASE_SALARY
    .transform('max') # 원래 index를 유지하는 메서드
)

max_sal.index

# In[106]:


(employee
    .assign(MAX_DEPT_SALARY=max_sal)
)


# In[107]:


max_sal = (employee
    .groupby('DEPARTMENT')
    .BASE_SALARY
    .max()
)

max_sal.index
employee.columns

# In[108]:


(employee
    .merge(max_sal.rename('MAX_DEPT_SALARY'),
           how='left', left_on='DEPARTMENT',
           right_index=True)
)


# ## Highlighting the maximum value from each column

# ### How to do it...

# In[9]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college.dtypes


# In[110]:


college.MD_EARN_WNE_P10.sample(10, random_state=42)


# In[111]:


college.GRAD_DEBT_MDN_SUPP.sample(10, random_state=42)


# In[112]:


college.MD_EARN_WNE_P10.value_counts()


# In[113]:


set(college.MD_EARN_WNE_P10.apply(type))


# In[114]:


college.GRAD_DEBT_MDN_SUPP.value_counts()


# In[115]:

# 수치형으로 변경
cols = ['MD_EARN_WNE_P10', 'GRAD_DEBT_MDN_SUPP']
for col in cols:
    college[col] = pd.to_numeric(college[col], errors='coerce')


# In[116]:


college.dtypes.loc[cols]


# In[11]:


college_n = college.select_dtypes('number')
college_n.head()


# In[13]:


binary_only = college_n.nunique() == 2
binary_only.head()


# In[14]:

# Binary only에서 True 컬럼만 추출 
binary_cols = binary_only[binary_only].index.tolist()
binary_cols


# In[15]:

# 이진열 삭제
college_n2 = college_n.drop(columns=binary_cols)
college_n2.head()


# In[16]:

# .idxmax 메소드: 각 열의 최대값의 인덱스를 추출
max_cols = college_n2.idxmax()
max_cols


# In[17]:


unique_max_cols = max_cols.unique()
unique_max_cols[:5]


# In[123]:


college_n2.loc[unique_max_cols] #.style.highlight_max()


# In[18]:


college_n2.loc[unique_max_cols].style.highlight_max()

# In[124]:

# 코드 정리
def remove_binary_cols(df):
    binary_only = df.nunique() == 2
    cols = binary_only[binary_only].index.tolist()
    return df.drop(columns=cols)


# In[125]:


def select_rows_with_max_cols(df):
    max_cols = df.idxmax()
    unique = max_cols.unique()
    return df.loc[unique]


# In[126]:


(college
   .assign(
       MD_EARN_WNE_P10=pd.to_numeric(college.MD_EARN_WNE_P10, errors='coerce'),
       GRAD_DEBT_MDN_SUPP=pd.to_numeric(college.GRAD_DEBT_MDN_SUPP, errors='coerce'))
   .select_dtypes('number')
   .pipe(remove_binary_cols)
   .pipe(select_rows_with_max_cols)
)


# ### How it works...

# ### There's more...

# In[19]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_').head()


# In[20]:


college_ugds.style.highlight_max(axis='columns')


# ## Replicating idxmax with method chaining

# ### How to do it...

# In[128]:


def remove_binary_cols(df):
    binary_only = df.nunique() == 2
    cols = binary_only[binary_only].index.tolist()
    return df.drop(columns=cols)


# In[129]:


college_n = (college
   .assign(
       MD_EARN_WNE_P10=pd.to_numeric(college.MD_EARN_WNE_P10, errors='coerce'),
       GRAD_DEBT_MDN_SUPP=pd.to_numeric(college.GRAD_DEBT_MDN_SUPP, errors='coerce'))
   .select_dtypes('number')
   .pipe(remove_binary_cols)
)


# In[130]:


college_n.max().head()


# In[131]:


college_n.eq(college_n.max()).head()


# In[132]:


has_row_max = (college_n
    .eq(college_n.max())
    .any(axis='columns')
)
has_row_max.head()


# In[133]:


college_n.shape


# In[134]:


has_row_max.sum()


# In[135]:


college_n.eq(college_n.max()).cumsum()


# In[136]:


(college_n
    .eq(college_n.max())
    .cumsum()
    .cumsum()
)


# In[137]:


has_row_max2 = (college_n
    .eq(college_n.max()) 
    .cumsum() 
    .cumsum() 
    .eq(1) 
    .any(axis='columns')
)


# In[138]:


has_row_max2.head()


# In[139]:


has_row_max2.sum()


# In[140]:


idxmax_cols = has_row_max2[has_row_max2].index
idxmax_cols


# In[141]:


set(college_n.idxmax().unique()) == set(idxmax_cols)


# In[142]:


def idx_max(df):
     has_row_max = (df
         .eq(df.max())
         .cumsum()
         .cumsum()
         .eq(1)
         .any(axis='columns')
     )
     return has_row_max[has_row_max].index


# In[143]:


idx_max(college_n)
len(idx_max(college_n))

len(college_n.idxmax().values)
# ### How it works...

# ### There's more...

# In[144]:


def idx_max(df):
     has_row_max = (df
         .eq(df.max())
         .cumsum()
         .cumsum()
         .eq(1)
         .any(axis='columns')
         [lambda df_: df_]
         .index
     )
     return has_row_max


# In[145]:


get_ipython().run_line_magic('timeit', 'college_n.idxmax().values')


# In[146]:


get_ipython().run_line_magic('timeit', 'idx_max(college_n)')


# ## Finding the most common maximum of columns

# ### How to do it...

# In[147]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
college_ugds = college.filter(like='UGDS_')
college_ugds.head()


# In[148]:


highest_percentage_race = college_ugds.idxmax(axis='columns')
highest_percentage_race.head()


# In[149]:


highest_percentage_race.value_counts(normalize=True)


# ### How it works...

# ### There's more...

# In[150]:

 

# In[ ]:




