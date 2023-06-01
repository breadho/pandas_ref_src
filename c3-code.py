#!/usr/bin/env python
# coding: utf-8

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")


# In[ ]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 4, 'max_rows', 10, 'max_colwidth', 12)


# ### How to do it\...

# In[ ]:


fname = ['Paul', 'John', 'Richard', 'George']
lname = ['McCartney', 'Lennon', 'Starkey', 'Harrison']
birth = [1942, 1940, 1940, 1943]


# In[ ]:


people = {'first': fname, 'last': lname, 'birth': birth}


# In[ ]:


beatles = pd.DataFrame(people)
beatles


# ### How it works\...

# In[ ]:


beatles.index


# In[ ]:


pd.DataFrame(people, index=['a', 'b', 'c', 'd'])


# ### There\'s More

# In[ ]:


pd.DataFrame(
[{"first":"Paul", "last":"McCartney", "birth":1942},
 {"first":"John", "last":"Lennon", "birth":1940},
 {"first":"Richard", "last":"Starkey", "birth":1940},
 {"first":"George", "last":"Harrison", "birth":1943}])


# In[ ]:

pd.DataFrame(
[{"first":"Paul","last":"McCartney", "birth":1942},
 {"first":"John","last":"Lennon", "birth":1940},
 {"first":"Richard","last":"Starkey", "birth":1940},
 {"first":"George","last":"Harrison", "birth":1943}],
 columns=['last', 'first', 'birth'])


# ### How to do it\...

# In[ ]:


beatles


# In[ ]:


from io import StringIO
fout = StringIO()
beatles.to_csv(fout)  # use a filename instead of fout
# beatles.to_csv("test.csv") 

# In[ ]:


print(fout.getvalue())


# ### There\'s More

# In[ ]:


_ = fout.seek(0)
pd.read_csv(fout)


# In[ ]:


_ = fout.seek(0)
pd.read_csv(fout, index_col=0)


# In[ ]:


fout = StringIO()
beatles.to_csv(fout, index=False) 
print(fout.getvalue())


# ### How to do it\...

# In[ ]:

os.listdir()

diamonds = pd.read_csv('data/diamonds.csv', nrows=1000)
diamonds


# In[ ]:


diamonds.info()


# In[ ]:


diamonds2 = pd.read_csv('data/diamonds.csv', 
                        nrows=1000,
                        dtype={'carat': np.float32, 
                               'depth': np.float32,
                               'table': np.float32, 
                               'x': np.float32,
                               'y': np.float32, 
                               'z': np.float32,
                               'price': np.int16})


# In[ ]:


diamonds2.info()


# In[ ]:


diamonds.describe()


# In[ ]:


diamonds2.describe()


# In[ ]:


diamonds2.cut.value_counts()


# In[ ]:


diamonds2.color.value_counts()


# In[ ]:


diamonds2.clarity.value_counts()


# In[ ]:


diamonds3 = pd.read_csv('data/diamonds.csv', 
                        nrows=1000,
                        dtype={'carat': np.float32, 
                               'depth': np.float32,
                               'table': np.float32, 
                                  'x': np.float32,
                                  'y': np.float32, 
                                  'z': np.float32,
                                  'price': np.int16,
                                  'cut': 'category', 
                                  'color': 'category',
                                  'clarity': 'category'})


# In[ ]:


diamonds3.info()


# In[ ]:


np.iinfo(np.int8)
np.iinfo(np.int16)
np.iinfo(np.int32)
np.iinfo(np.int64)


# In[ ]:


np.finfo(np.float16)
np.finfo(np.float32)
np.finfo(np.float64)


# In[ ]:

# col_selection
cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']

diamonds4 = pd.read_csv('data/diamonds.csv', 
                        nrows=1000,
                        dtype={'carat': np.float32, 
                               'depth': np.float32,
                               'table': np.float32, 
                               'price': np.int16,
                               'cut': 'category', 
                               'color': 'category',
                               'clarity': 'category'},
                        usecols = cols)

diamonds4.dtypes

# In[ ]:


diamonds4.info()


# In[ ]:

# Chunk 이용
cols = ['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'price']

diamonds_iter = pd.read_csv('data/diamonds.csv', 
                            nrows=1000,
                            dtype={'carat': np.float32, 
                                   'depth': np.float32,
                                   'table': np.float32, 
                                   'price': np.int16,
                                   'cut': 'category', 
                                   'color': 'category',
                                   'clarity': 'category'},
                            usecols = cols,
                            chunksize = 200)


# In[ ]:


def process(df):
    return f'processed {df.size} items'


# In[ ]:


for chunk in diamonds_iter:
    process(chunk)


# ### How it works\...

# ### There\'s more \...

# In[ ]:

# Return the memory usage of each column in bytes
diamonds.price.memory_usage()

# In[ ]:


diamonds.price.memory_usage(index=False)


# In[ ]:


diamonds.cut.memory_usage()


# In[ ]:


diamonds.cut.memory_usage(deep=True)



# In[ ]:

# Feather 형식과 같이 형식을 추적하는 이진 형식으로 저장할 수 있음

# read_feather() reads a Feather file as a pandas.DataFrame. 
# read_table() reads a Feather file as a Table. Internally, 
# read_feather() simply calls read_table() and the result is converted to pandas:

import pyarrow.feather as feather
diamonds4.to_feather('tmp/d.arr')
diamonds5 = pd.read_feather('tmp/d.arr')

diamonds5.info() # 17.7 MB

# os.mkdir('tmp')
# os.listdir()


# In[ ]:


diamonds4.to_parquet('tmp/d.pqt')
diamonds6 = pd.read_parquet('tmp/d.pqt')
diamonds6.info()


# ### How to do it\...

# In[ ]:


beatles.to_excel('tmp/beat.xls')


# In[ ]:


beatles.to_excel('tmp/beat.xlsx')


# In[ ]:


beat2 = pd.read_excel('tmp/beat.xls')
beat2


# In[ ]:


beat2 = pd.read_excel('tmp/beat.xls', index_col=0)
beat2


# In[ ]:


beat2.dtypes


# ### How it works\...

# ### There\'s more\...

# In[ ]:

# 판다스 이용 스프레드 시트 내 특정 시트 작성

xl_writer = pd.ExcelWriter('tmp/beat.xlsx')
beatles.to_excel(xl_writer, sheet_name='All')
beatles[beatles.birth < 1941].to_excel(xl_writer, sheet_name = '1940')
xl_writer.save()



# ### How to do it\...

# In[ ]:


autos = pd.read_csv('data/vehicles.csv.zip')
autos


# In[ ]:


autos.modifiedOn.dtype


# In[ ]:


autos.modifiedOn


# In[ ]:


pd.to_datetime(autos.modifiedOn)  # doctest: +SKIP


# In[ ]:


autos = pd.read_csv('data/vehicles.csv.zip',
    parse_dates=['modifiedOn'])  # doctest: +SKIP
autos.modifiedOn


# In[ ]:


import zipfile


# In[ ]:


with zipfile.ZipFile('data/kaggle-survey-2018.zip') as z:
    print('\n'.join(z.namelist()))
    kag = pd.read_csv(z.open('multipleChoiceResponses.csv'))
    kag_questions = kag.iloc[0]
    survey = kag.iloc[1:]


# In[ ]:


print(survey.head(2).T)


# ### How it works\...

# ### There\'s more\...

# ### How to do it\...

# In[ ]:


import sqlite3

con = sqlite3.connect('data/beat.db')

with con:
    cur = con.cursor()
    cur.execute("""DROP TABLE Band""")
    cur.execute("""CREATE TABLE Band(id INTEGER PRIMARY KEY,
        fname TEXT, lname TEXT, birthyear INT)""")
    cur.execute("""INSERT INTO Band VALUES(
        0, 'Paul', 'McCartney', 1942)""")
    cur.execute("""INSERT INTO Band VALUES(
        1, 'John', 'Lennon', 1940)""")
    _ = con.commit()


# In[ ]:


import sqlalchemy as sa

engine = sa.create_engine(
  'sqlite:///data/beat.db', echo=True)

sa_connection = engine.connect()


# In[ ]:


beat = pd.read_sql('Band', sa_connection, index_col='id')
beat


# In[ ]:


sql = '''SELECT fname, birthyear from Band'''
fnames = pd.read_sql(sql, con)
fnames


# ### How it work\'s\...

# In[ ]:


import json
encoded = json.dumps(people)
encoded


# In[ ]:


json.loads(encoded)


# ### How to do it\...

# In[ ]:


beatles = pd.read_json(encoded)
beatles


# In[ ]:


records = beatles.to_json(orient='records')
records


# In[ ]:


pd.read_json(records, orient='records')


# In[ ]:


split = beatles.to_json(orient='split')
split


# In[ ]:


pd.read_json(split, orient='split')


# In[ ]:


index = beatles.to_json(orient='index')
index


# In[ ]:


pd.read_json(index, orient='index')


# In[ ]:


values = beatles.to_json(orient='values')
values


# In[ ]:


pd.read_json(values, orient='values')


# In[ ]:


(pd.read_json(values, orient='values')
   .rename(columns=dict(enumerate(['first', 'last', 'birth'])))
)


# In[ ]:


table = beatles.to_json(orient='table')
table


# In[ ]:


pd.read_json(table, orient='table')


# ### How it works\...

# ### There\'s more\...

# In[ ]:

# 웹 서비스 작업 중이고 JSON에 데이터를 추가해야 하는 상황
# .to_dict 메서드 이용 딕셔너리 생성
output = beat.to_dict()
output


# In[ ]:


output['version'] = '0.4.1'
json.dumps(output)


# ### How to do it\...

# In[ ]:


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url)
len(dfs)


# In[ ]:


dfs[0]


# In[ ]:


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url, match='List of studio albums', na_values='—')
len(dfs)


# In[ ]:


dfs[0].columns


# In[ ]:


url ='https://en.wikipedia.org/wiki/The_Beatles_discography'
dfs = pd.read_html(url, match='List of studio albums', na_values='—',
    header=[0,1])
len(dfs)


# In[ ]:


dfs[0]


# In[ ]:


dfs[0].columns


# In[ ]:


df = dfs[0]
df.columns = ['Title', 'Release', 'UK', 'AUS', 'CAN', 'FRA', 'GER',
    'NOR', 'US', 'Certifications']
df


# In[ ]:


res = (df
  .pipe(lambda df_: df_[~df_.Title.str.startswith('Released')])
  .iloc[:-1]
  .assign(release_date=lambda df_: pd.to_datetime(
             df_.Release.str.extract(r'Released: (.*) Label')
               [0]
               .str.replace(r'\[E\]', '')
          ),
          label=lambda df_:df_.Release.str.extract(r'Label: (.*)')
         )
   .loc[:, ['Title', 'UK', 'AUS', 'CAN', 'FRA', 'GER', 'NOR',
            'US', 'release_date', 'label']]
)
res


# ### How it works\...

# ### There is more\...

# In[ ]:


url = 'https://github.com/mattharrison/datasets/blob/master/data/anscombes.csv'
dfs = pd.read_html(url, attrs={'class': 'csv-data'})
len(dfs)


# In[ ]:


dfs[0]


# In[ ]:




