#!/usr/bin/env python
# coding: utf-8

# ## Combining Pandas Objects


import os
os.getcwd()
os.chdir("D:/pandas_cookbook")


# In[1]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 7,'display.expand_frame_repr', True, # 'max_rows', 10, 
    'max_colwidth', 9, 'max_rows', 10, #'precision', 2
)#, 'width', 45)
pd.set_option('display.width', 65)


# ## Introduction

# ## Appending new rows to DataFrames

# ### How to do it...

# In[2]:


names = pd.read_csv('data/names.csv')
names


# In[3]:


new_data_list = ['Aria', 1]
names.loc[4] = new_data_list
names


# In[4]:


names.loc['five'] = ['Zach', 3]
names


# In[5]:


names.loc[len(names)] = {'Name':'Zayd', 'Age':2}
names


# In[6]:


names.loc[len(names)] = pd.Series({'Age':32, 'Name':'Dean'})
names


# In[7]:


names = pd.read_csv('data/names.csv')
names.append({'Name':'Aria', 'Age':1})


# In[8]:


names.append({'Name':'Aria', 'Age':1}, ignore_index=True)


# In[9]:


names.index = ['Canada', 'Canada', 'USA', 'USA']
names
 

# In[10]:


s = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s


# In[11]:


names.append(s)


# In[12]:


s1 = pd.Series({'Name': 'Zach', 'Age': 3}, name=len(names))
s2 = pd.Series({'Name': 'Zayd', 'Age': 2}, name='USA')
names.append([s1, s2])


# In[13]:


bball_16 = pd.read_csv('data/baseball16.csv')
bball_16


# In[14]:

# 단일 행을 Series로 선택, .to_dict 메서드를 체인시켜 예제 행을 딕셔너리 형태로 가져옴

data_dict = bball_16.iloc[0].to_dict()
data_dict


# In[15]:

# 이전 문자열 값을 모두 빈 문자열로 지정해 지우고, 다른 것은 결측치로 
# 지정하는 딕셔너리 컴프리헨션(dictionary comprehension) 할당

new_data_dict = {k: '' if isinstance(v, str) else
    np.nan for k, v in data_dict.items()}
new_data_dict


# ### How it works...

# ### There's more...

# In[16]:


random_data = []
for i in range(1000):   # doctest: +SKIP
    d = dict()
    for k, v in data_dict.items():
        if isinstance(v, str):
            d[k] = np.random.choice(list('abcde'))
        else:
            d[k] = np.random.randint(10)
    random_data.append(pd.Series(d, name=i + len(bball_16)))
random_data[0]


# ## Concatenating multiple DataFrames together

%%timeit 
bball_16_copy = bball_16.copy()
for row in random_data:
    bball_16_copy = bball_16_copy.append(row)



# ### How to do it...

# In[17]:

#  여러 데이터 프레임을 함께 연결
# concat 함수를 사용하면 두 개 이상의 데이터 프레임을 세로와 가로로 함께 연결할 수 있음
'''
평소와 마찬가지로 여러 pandas 객체를 동시에 처리하는 경우 연결은 
우연히 발생하는 것이 아니라 각 객체를 인덱스별로 정렬함
'''

stocks_2016 = pd.read_csv('data/stocks_2016.csv',
    index_col='Symbol')
stocks_2017 = pd.read_csv('data/stocks_2017.csv',
    index_col='Symbol')


# In[18]:


stocks_2016


# In[19]:


stocks_2017


# In[20]:


s_list = [stocks_2016, stocks_2017]
pd.concat(s_list)


# In[21]:


pd.concat(s_list, keys=['2016', '2017'], names=['Year', 'Symbol'])  


# In[22]:


pd.concat(s_list, keys=['2016', '2017'],
    axis='columns', names=['Year', None])    


# In[23]:

# join 방식: 기본 -> outer join
#            'inner' -> inner join

pd.concat(s_list, join='inner', keys=['2016', '2017'],
    axis='columns', names=['Year', None])


# ### How it works...

# ### There's more...

# In[24]:

# .append 함수는 DataFrame에 새 행만 추가할 수 있는 상당히 압축된 버전의 concat
# 내부적으로는 .append는 concat 함수를 호출함

stocks_2016.append(stocks_2017)


# ## Understanding the differences between concat, join, and merge

# ### How to do it...

# In[25]:
'''
concat, join, merge의 차이점 이해
'''

from IPython.display import display_html
years = 2016, 2017, 2018
stock_tables = [pd.read_csv(
    'data/stocks_{}.csv'.format(year), index_col = 'Symbol')
    for year in years]
stocks_2016, stocks_2017, stocks_2018 = stock_tables
stocks_2016


# In[26]:


stocks_2017


# In[27]:


stocks_2018


# In[28]:


pd.concat(stock_tables, keys=[2016, 2017, 2018])


# In[29]:

# axis 매개 변수를 columns로 변경하면 DataFrame을 수평으로 병합할 수 있음
pd.concat(dict(zip(years, stock_tables)), axis='columns')


# In[30]:


stocks_2016.join(stocks_2017, lsuffix='_2016',
    rsuffix='_2017', how='outer')


# In[31]:


other = [stocks_2017.add_suffix('_2017'),
    stocks_2018.add_suffix('_2018')]
stocks_2016.add_suffix('_2016').join(other, how='outer')


# In[32]:


stock_join = stocks_2016.add_suffix('_2016').join(other,
    how='outer')
stock_concat = pd.concat(dict(zip(years,stock_tables)),
    axis='columns')
level_1 = stock_concat.columns.get_level_values(1)
level_0 = stock_concat.columns.get_level_values(0).astype(str)
stock_concat.columns = level_1 + '_' + level_0
stock_join.equals(stock_concat)


# In[33]:


stocks_2016.merge(stocks_2017, left_index=True,
    right_index=True)


# In[34]:


step1 = stocks_2016.merge(stocks_2017, 
                          left_index=True,
                          right_index=True, 
                          how='outer',
                          suffixes=('_2016', '_2017'))

stock_merge = step1.merge(stocks_2018.add_suffix('_2018'),
                          left_index=True, 
                          right_index=True,
                          how='outer')

stock_concat.equals(stock_merge)


# In[35]:

# 인덱스나 열 레이블 자체가 아닌 열 값에 따라 정렬하는 경우 비교

names = ['prices', 'transactions']
food_tables = [pd.read_csv('data/food_{}.csv'.format(name))
    for name in names]
food_prices, food_transactions = food_tables
food_prices


# In[36]:


food_transactions


# In[37]:


food_transactions.merge(food_prices, on=['item', 'store'])    


# In[38]:


food_transactions.merge(food_prices.query('Date == 2017'), how='left')


# In[39]:


food_prices_join = food_prices.query('Date == 2017').set_index(['item', 'store'])
food_prices_join    


# In[40]:


food_transactions.join(food_prices_join, on=['item', 'store'])


# In[41]:


pd.concat([food_transactions.set_index(['item', 'store']),
           food_prices.set_index(['item', 'store'])],
          axis='columns')


# ### How it works...

# ### There's more...

# In[42]:


import glob
df_list = []
for filename in glob.glob('data/gas prices/*.csv'):
    df_list.append(pd.read_csv(filename, 
                               index_col = 'Week',
                               parse_dates = ['Week']))
gas = pd.concat(df_list, axis='columns')
gas


# ## Connecting to SQL databases

# ### How to do it...

# In[43]:


from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/chinook.db')


# In[44]:


tracks = pd.read_sql_table('tracks', engine)
tracks


# In[45]:


(pd.read_sql_table('genres', engine)
     .merge(tracks[['GenreId', 'Milliseconds']],
            on='GenreId', how='left') 
     .drop('GenreId', axis='columns')
)


# In[46]:


(pd.read_sql_table('genres', engine)
     .merge(tracks[['GenreId', 'Milliseconds']],
            on='GenreId', how='left') 
     .drop('GenreId', axis='columns')
     .groupby('Name')
     ['Milliseconds']
     .mean()
     .pipe(lambda s_: pd.to_timedelta(s_, unit='ms'))
     .dt.floor('s')
     .sort_values()
)


# In[47]:


cust = pd.read_sql_table('customers', engine,
    columns=['CustomerId','FirstName',
    'LastName'])
invoice = pd.read_sql_table('invoices', engine,
    columns=['InvoiceId','CustomerId'])
ii = pd.read_sql_table('invoice_items', engine,
    columns=['InvoiceId', 'UnitPrice', 'Quantity'])
(cust
    .merge(invoice, on='CustomerId') 
    .merge(ii, on='InvoiceId')
)


# In[48]:


(cust
    .merge(invoice, on='CustomerId') 
    .merge(ii, on='InvoiceId')
    .assign(Total=lambda df_:df_.Quantity * df_.UnitPrice)
    .groupby(['CustomerId', 'FirstName', 'LastName'])
    ['Total']
    .sum()
    .sort_values(ascending=False) 
)


# ### How it works...

# ### There's more...

# In[49]:


sql_string1 = '''
SELECT
    Name,
    time(avg(Milliseconds) / 1000, 'unixepoch') as avg_time
FROM (
      SELECT
          g.Name,
          t.Milliseconds
      FROM
          genres as g
      JOIN
          tracks as t on
          g.genreid == t.genreid
     )
GROUP BY Name
ORDER BY avg_time'''
pd.read_sql_query(sql_string1, engine)


# In[50]:


sql_string2 = '''
   SELECT
         c.customerid,
         c.FirstName,
         c.LastName,
         sum(ii.quantity * ii.unitprice) as Total
   FROM
        customers as c
   JOIN
        invoices as i
        on c.customerid = i.customerid
   JOIN
       invoice_items as ii
       on i.invoiceid = ii.invoiceid
   GROUP BY
       c.customerid, c.FirstName, c.LastName
   ORDER BY
       Total desc'''


# In[51]:


pd.read_sql_query(sql_string2, engine)


# In[ ]:

# SQL 연습

from sqlalchemy import create_engine
engine = create_engine('sqlite:///data/chinook.db')

tracks = pd.read_sql_table('tracks', engine)

(
 pd.read_sql_table('genres', engine)
 .merge(tracks[['GenreId', 'Milliseconds']], on = 'GenreId', how = 'left')
 .drop('GenreId', axis = 'columns')
)


(
 pd.read_sql_table('genres', engine)
 .merge(tracks[['GenreId', 'Milliseconds']], on = 'GenreId', how = 'left')
 .drop('GenreId', axis = 'columns')
 .groupby('Name')
 ['Milliseconds']
 .mean()
 .pipe(lambda s_: pd.to_timedelta(s_, unit = 'ms').rename('Length')
       )
 .dt.floor('s')
 .sort_values()
)

# 고객당 총 지출 추출
cust = pd.read_sql_table('customers', engine,
                         columns = ['CustomerId', 'FirstName', 'LastName'])

cust

invoice = pd.read_sql_table('invoices', engine, 
                            columns = ['InvoiceId', 'CustomerId'])

invoice_items = pd.read_sql_table('invoice_items', 
                                  engine, 
                                  columns = ['InvoiceId', 'UnitPrice', 'Quantity'])

(
 cust.merge(invoice, on = 'CustomerId')
 .merge(invoice_items, on = 'InvoiceId')
)

# 수량과 단위 가격을 곱하면 고객당 총지출을 구할 수 있음
(cust
 .merge(invoice, on = 'CustomerId')
 .merge(invoice_items, on = 'InvoiceId')
 .assign(Total = lambda df_: df_.Quantity*df_.UnitPrice)
 .groupby(['CustomerId', 'FirstName', 'LastName'])
 ['Total']
 .sum()
 .sort_values(ascending = False)
)








