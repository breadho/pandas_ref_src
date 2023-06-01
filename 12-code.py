#!/usr/bin/env python
# coding: utf-8

# # Time Series Analysis

# In[10]:


import pandas as pd
import numpy as np
pd.set_option('max_columns', 7,'display.expand_frame_repr', True, # 'max_rows', 10, 
    'max_colwidth', 12, 'max_rows', 10, #'precision', 2
)#, 'width', 45)
pd.set_option('display.width', 65)


# In[ ]:


pd.set_option(#'max_columns', 4,
    'max_rows', 10)
from io import StringIO
def txt_repr(df, width=40, rows=None):
    buf = StringIO()
    rows = rows if rows is not None else pd.options.display.max_rows
    num_cols = len(df.columns)
    with pd.option_context('display.width', 100):
        df.to_string(buf=buf, max_cols=num_cols, max_rows=rows,line_width=width)
        out = buf.getvalue()
        for line in out.split('\n'):
            if len(line) > width or line.strip().endswith('\\'):
                break
        else:
            return out
        done = False
        while not done:
            buf = StringIO()
            df.to_string(buf=buf, max_cols=num_cols, max_rows=rows,line_width=width)
            for line in buf.getvalue().split('\n'):
                if line.strip().endswith('\\'):
                    num_cols = min([num_cols - 1, int(num_cols*.8)])
                    break
            else:
                break
        return buf.getvalue()
pd.DataFrame.__repr__ = lambda self, *args: txt_repr(self, 65, 10)


# ## Introduction

# ## Understanding the difference between Python and pandas date tools

# ### How to do it...

# In[11]:


import datetime
date = datetime.date(year=2013, month=6, day=7)
time = datetime.time(hour=12, minute=30,
    second=19, microsecond=463198)
dt = datetime.datetime(year=2013, month=6, day=7,
    hour=12, minute=30, second=19,
    microsecond=463198)
print(f"date is {date}")


# In[12]:


print(f"time is {time}")


# In[13]:


print(f"datetime is {dt}")


# In[14]:


td = datetime.timedelta(weeks=2, days=5, hours=10,
    minutes=20, seconds=6.73,
    milliseconds=99, microseconds=8)
td


# In[15]:


print(f'new date is {date+td}')


# In[16]:


print(f'new datetime is {dt+td}')


# In[17]:


time + td


# In[156]:


pd.Timestamp(year=2012, month=12, day=21, hour=5,
   minute=10, second=8, microsecond=99)


# In[157]:


pd.Timestamp('2016/1/10')


# In[158]:


pd.Timestamp('2014-5/10')


# In[159]:


pd.Timestamp('Jan 3, 2019 20:45.56')


# In[160]:


pd.Timestamp('2016-01-05T05:34:43.123456789')


# In[161]:


pd.Timestamp(500)


# In[162]:


pd.Timestamp(5000, unit='D')


# In[163]:


pd.to_datetime('2015-5-13')


# In[164]:


pd.to_datetime('2015-13-5', dayfirst=True)


# In[165]:


pd.to_datetime('Start Date: Sep 30, 2017 Start Time: 1:30 pm',
    format='Start Date: %b %d, %Y Start Time: %I:%M %p')


# In[166]:


pd.to_datetime(100, unit='D', origin='2013-1-1')


# In[167]:


s = pd.Series([10, 100, 1000, 10000])
pd.to_datetime(s, unit='D')


# In[168]:


s = pd.Series(['12-5-2015', '14-1-2013',
   '20/12/2017', '40/23/2017'])


# In[169]:


pd.to_datetime(s, dayfirst=True, errors='coerce')


# In[170]:


pd.to_datetime(['Aug 3 1999 3:45:56', '10/31/2017'])


# In[171]:


pd.Timedelta('12 days 5 hours 3 minutes 123456789 nanoseconds')


# In[172]:


pd.Timedelta(days=5, minutes=7.34)


# In[173]:


pd.Timedelta(100, unit='W')


# In[174]:


pd.to_timedelta('67:15:45.454')


# In[175]:


s = pd.Series([10, 100])
pd.to_timedelta(s, unit='s')


# In[176]:


time_strings = ['2 days 24 minutes 89.67 seconds',
    '00:45:23.6']
pd.to_timedelta(time_strings)


# In[177]:


pd.Timedelta('12 days 5 hours 3 minutes') * 2


# In[178]:


(pd.Timestamp('1/1/2017') + 
   pd.Timedelta('12 days 5 hours 3 minutes') * 2)


# In[179]:


td1 = pd.to_timedelta([10, 100], unit='s')
td2 = pd.to_timedelta(['3 hours', '4 hours'])
td1 + td2


# In[180]:


pd.Timedelta('12 days') / pd.Timedelta('3 days')


# In[181]:


ts = pd.Timestamp('2016-10-1 4:23:23.9')
ts.ceil('h')


# In[182]:


ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second


# In[183]:


ts.dayofweek, ts.dayofyear, ts.daysinmonth


# In[184]:


ts.to_pydatetime()


# In[185]:


td = pd.Timedelta(125.8723, unit='h')
td


# In[186]:


td.round('min')


# In[187]:


td.components


# In[188]:


td.total_seconds()


# ### How it works...

# ### There's more...

# In[189]:


date_string_list = ['Sep 30 1984'] * 10000
get_ipython().run_line_magic('timeit', "pd.to_datetime(date_string_list, format='%b %d %Y')")


# In[190]:


get_ipython().run_line_magic('timeit', 'pd.to_datetime(date_string_list)')


# ## Slicing time series intelligently

# ### How to do it...

# In[191]:


crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes


# In[192]:


crime = crime.set_index('REPORTED_DATE')
crime


# In[193]:


crime.loc['2016-05-12 16:45:00']


# In[194]:


crime.loc['2016-05-12']


# In[195]:


crime.loc['2016-05'].shape


# In[196]:


crime.loc['2016'].shape


# In[197]:


crime.loc['2016-05-12 03'].shape


# In[198]:


crime.loc['Dec 2015'].sort_index()


# In[199]:


crime.loc['2016 Sep, 15'].shape


# In[200]:


crime.loc['21st October 2014 05'].shape


# In[201]:


crime.loc['2015-3-4':'2016-1-1'].sort_index()


# In[202]:


crime.loc['2015-3-4 22':'2016-1-1 11:22:00'].sort_index()


# ### How it works...

# In[203]:


mem_cat = crime.memory_usage().sum()
mem_obj = (crime
   .astype({'OFFENSE_TYPE_ID':'object',
            'OFFENSE_CATEGORY_ID':'object',
           'NEIGHBORHOOD_ID':'object'}) 
   .memory_usage(deep=True)
   .sum()
)
mb = 2 ** 20
round(mem_cat / mb, 1), round(mem_obj / mb, 1)


# In[204]:


crime.index[:2]


# ### There's more...

# In[205]:


get_ipython().run_line_magic('timeit', "crime.loc['2015-3-4':'2016-1-1']")


# In[206]:


crime_sort = crime.sort_index()
get_ipython().run_line_magic('timeit', "crime_sort.loc['2015-3-4':'2016-1-1']")


# ## Filtering columns with time data

# ### How to do it... 

# In[207]:


crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes


# In[208]:


(crime
    [crime.REPORTED_DATE == '2016-05-12 16:45:00']
)


# In[209]:


(crime
    [crime.REPORTED_DATE == '2016-05-12']
)


# In[210]:


(crime
    [crime.REPORTED_DATE.dt.date == '2016-05-12']
)


# In[211]:


(crime
    [crime.REPORTED_DATE.between(
         '2016-05-12', '2016-05-13')]
)


# In[212]:


(crime
    [crime.REPORTED_DATE.between(
         '2016-05', '2016-06')]
    .shape
)


# In[213]:


(crime
    [crime.REPORTED_DATE.between(
         '2016', '2017')]
    .shape
)


# In[214]:


(crime
    [crime.REPORTED_DATE.between(
         '2016-05-12 03', '2016-05-12 04')]
    .shape
)


# In[215]:


(crime
    [crime.REPORTED_DATE.between(
         '2016 Sep, 15', '2016 Sep, 16')]
    .shape
)


# In[216]:


(crime
    [crime.REPORTED_DATE.between(
         '21st October 2014 05', '21st October 2014 06')]
    .shape
)


# In[217]:


(crime
    [crime.REPORTED_DATE.between(
         '2015-3-4 22','2016-1-1 23:59:59')]
    .shape
)


# In[218]:


(crime
    [crime.REPORTED_DATE.between(
         '2015-3-4 22','2016-1-1 11:22:00')]
    .shape
)


# ### How it works...

# In[219]:


lmask = crime.REPORTED_DATE >= '2015-3-4 22'
rmask = crime.REPORTED_DATE <= '2016-1-1 11:22:00'
crime[lmask & rmask].shape


# ### There's more...

# In[220]:


ctseries = crime.set_index('REPORTED_DATE')
get_ipython().run_line_magic('timeit', "ctseries.loc['2015-3-4':'2016-1-1']")


# In[221]:


get_ipython().run_line_magic('timeit', "crime[crime.REPORTED_DATE.between('2015-3-4','2016-1-1')]")


# ## Using methods that only work with a DatetimeIndex

# ### How to do it...

# In[222]:


crime = (pd.read_hdf('data/crime.h5', 'crime') 
    .set_index('REPORTED_DATE')
)
type(crime.index)


# In[223]:


crime.between_time('2:00', '5:00', include_end=False)


# In[224]:


crime.at_time('5:47')


# In[225]:


crime_sort = crime.sort_index()
crime_sort.first(pd.offsets.MonthBegin(6))


# In[226]:


crime_sort.first(pd.offsets.MonthEnd(6))


# In[227]:


crime_sort.first(pd.offsets.MonthBegin(6, normalize=True))


# In[228]:


crime_sort.loc[:'2012-06']


# In[229]:


crime_sort.first('5D') # 5 days


# In[230]:


crime_sort.first('5B') # 5 business days


# In[231]:


crime_sort.first('7W') # 7 weeks, with weeks ending on Sunday


# In[232]:


crime_sort.first('3QS') # 3rd quarter start


# In[233]:


crime_sort.first('A') # one year end


# ### How it works...

# In[234]:


import datetime
crime.between_time(datetime.time(2,0), datetime.time(5,0),
                   include_end=False)


# In[235]:


first_date = crime_sort.index[0]
first_date


# In[236]:


first_date + pd.offsets.MonthBegin(6)


# In[237]:


first_date + pd.offsets.MonthEnd(6)


# In[238]:


step4 = crime_sort.first(pd.offsets.MonthEnd(6))
end_dt = crime_sort.index[0] + pd.offsets.MonthEnd(6)
step4_internal = crime_sort[:end_dt]
step4.equals(step4_internal)


# ### There's more...

# In[239]:


dt = pd.Timestamp('2012-1-16 13:40')
dt + pd.DateOffset(months=1)


# In[240]:


do = pd.DateOffset(years=2, months=5, days=3,
    hours=8, seconds=10)
pd.Timestamp('2012-1-22 03:22') + do


# ## Counting the number of weekly crimes

# ### How to do it...

# In[241]:


crime_sort = (pd.read_hdf('data/crime.h5', 'crime') 
    .set_index('REPORTED_DATE') 
    .sort_index()
)


# In[242]:


crime_sort.resample('W')


# In[243]:


(crime_sort
    .resample('W')
    .size()
)


# In[244]:


len(crime_sort.loc[:'2012-1-8'])


# In[245]:


len(crime_sort.loc['2012-1-9':'2012-1-15'])


# In[246]:


(crime_sort
    .resample('W-THU')
    .size()
)


# In[247]:


weekly_crimes = (crime_sort
    .groupby(pd.Grouper(freq='W')) 
    .size()
)
weekly_crimes


# ### How it works...

# In[248]:


r = crime_sort.resample('W')
[attr for attr in dir(r) if attr[0].islower()]


# ### There's more...

# In[249]:


crime = pd.read_hdf('data/crime.h5', 'crime')
weekly_crimes2 = crime.resample('W', on='REPORTED_DATE').size()
weekly_crimes2.equals(weekly_crimes)


# In[250]:


weekly_crimes_gby2 = (crime
    .groupby(pd.Grouper(key='REPORTED_DATE', freq='W'))
    .size()
)
weekly_crimes2.equals(weekly_crimes)


# In[251]:


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(16, 4))
weekly_crimes.plot(title='All Denver Crimes', ax=ax)
fig.savefig('/tmp/c12-crimes.png', dpi=300)


# ## Aggregating weekly crime and traffic accidents separately

# ### How to do it...

# In[252]:


crime = (pd.read_hdf('data/crime.h5', 'crime') 
    .set_index('REPORTED_DATE') 
    .sort_index()
)


# In[253]:


(crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


# In[254]:


(crime
    .resample('QS')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


# In[255]:


(crime
   .loc['2012-4-1':'2012-6-30', ['IS_CRIME', 'IS_TRAFFIC']]
   .sum()
)


# In[256]:


(crime
    .groupby(pd.Grouper(freq='Q')) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
)


# In[257]:


fig, ax = plt.subplots(figsize=(16, 4))
(crime
    .groupby(pd.Grouper(freq='Q')) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .plot(color=['black', 'lightgrey'], ax=ax,
          title='Denver Crimes and Traffic Accidents')
)
fig.savefig('/tmp/c12-crimes2.png', dpi=300)


# ### How it works...

# In[258]:


(crime
    .resample('Q')
    .sum()
)


# In[259]:


(crime_sort
    .resample('QS-MAR')
    ['IS_CRIME', 'IS_TRAFFIC'] 
    .sum()
)


# ### There's more...

# In[260]:


crime_begin = (crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .iloc[0]
)


# In[261]:


fig, ax = plt.subplots(figsize=(16, 4))
(crime
    .resample('Q')
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .div(crime_begin)
    .sub(1)
    .round(2)
    .mul(100)
    .plot.bar(color=['black', 'lightgrey'], ax=ax,
          title='Denver Crimes and Traffic Accidents % Increase')
)


# In[262]:


fig.autofmt_xdate()
fig.savefig('/tmp/c12-crimes3.png', dpi=300, bbox_inches='tight')


# ## Measuring crime by weekday and year

# ### How to do it...

# In[263]:


crime = pd.read_hdf('data/crime.h5', 'crime')
crime


# In[264]:


(crime
   ['REPORTED_DATE']
   .dt.weekday_name 
   .value_counts()
)


# In[265]:


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday']
title = 'Denver Crimes and Traffic Accidents per Weekday'
fig, ax = plt.subplots(figsize=(6, 4))
(crime
   ['REPORTED_DATE']
   .dt.weekday_name 
   .value_counts()
   .reindex(days)
   .plot.barh(title=title, ax=ax)
)
fig.savefig('/tmp/c12-crimes4.png', dpi=300, bbox_inches='tight')                 


# In[266]:


title = 'Denver Crimes and Traffic Accidents per Year'
fig, ax = plt.subplots(figsize=(6, 4))
(crime
   ['REPORTED_DATE']
   .dt.year 
   .value_counts()
   .plot.barh(title=title, ax=ax)
)
fig.savefig('/tmp/c12-crimes5.png', dpi=300, bbox_inches='tight')                 


# In[267]:


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
)


# In[268]:


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
)


# In[269]:


criteria = crime['REPORTED_DATE'].dt.year == 2017
crime.loc[criteria, 'REPORTED_DATE'].dt.dayofyear.max()


# In[270]:


round(272 / 365, 3)


# In[271]:


crime_pct = (crime
   ['REPORTED_DATE']
   .dt.dayofyear.le(272) 
   .groupby(crime.REPORTED_DATE.dt.year) 
   .mean()
   .round(3)
)


# In[272]:


crime_pct


# In[273]:


crime_pct.loc[2012:2016].median()


# In[274]:


def update_2017(df_):
    df_.loc[2017] = (df_
        .loc[2017]
        .div(.748) 
        .astype('int')
    )
    return df_
(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
)


# In[275]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
table = (crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
)
sns.heatmap(table, cmap='Greys', ax=ax)
fig.savefig('/tmp/c12-crimes6.png', dpi=300, bbox_inches='tight')                 


# In[276]:


denver_pop = pd.read_csv('data/denver_pop.csv',
    index_col='Year')
denver_pop


# In[277]:


den_100k = denver_pop.div(100_000).squeeze()
normalized = (crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
    .div(den_100k, axis='index')
    .astype(int)
)
normalized


# In[278]:


import seaborn as sns
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(normalized, cmap='Greys', ax=ax)
fig.savefig('/tmp/c12-crimes7.png', dpi=300, bbox_inches='tight')                 


# ### How it works...

# In[279]:


(crime
   ['REPORTED_DATE']
   .dt.weekday_name 
   .value_counts()
   .loc[days]
)


# In[280]:


(crime
    .assign(year=crime.REPORTED_DATE.dt.year,
            day=crime.REPORTED_DATE.dt.weekday_name)
    .pipe(lambda df_: pd.crosstab(df_.year, df_.day))
)


# In[281]:


(crime
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
) / den_100k


# ### There's more...

# In[282]:


days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
        'Friday', 'Saturday', 'Sunday']
crime_type = 'auto-theft'
normalized = (crime
    .query('OFFENSE_CATEGORY_ID == @crime_type')
    .groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
              crime['REPORTED_DATE'].dt.weekday_name.rename('day')])
    .size()
    .unstack('day')
    .pipe(update_2017)
    .reindex(columns=days)
    .div(den_100k, axis='index')
    .astype(int)
)
normalized


# ## Grouping with anonymous functions with a DatetimeIndex

# ### How to do it...

# In[283]:


crime = (pd.read_hdf('data/crime.h5', 'crime') 
   .set_index('REPORTED_DATE') 
   .sort_index()
)


# In[284]:


common_attrs = (set(dir(crime.index)) & 
    set(dir(pd.Timestamp)))
[attr for attr in common_attrs if attr[0] != '_']


# In[285]:


crime.index.weekday_name.value_counts()


# In[286]:


(crime
   .groupby(lambda idx: idx.weekday_name) 
   ['IS_CRIME', 'IS_TRAFFIC']
   .sum()    
)


# In[287]:


funcs = [lambda idx: idx.round('2h').hour, lambda idx: idx.year]
(crime
    .groupby(funcs) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .unstack()
)


# In[288]:


funcs = [lambda idx: idx.round('2h').hour, lambda idx: idx.year]
(crime
    .groupby(funcs) 
    ['IS_CRIME', 'IS_TRAFFIC']
    .sum()
    .unstack()
    .style.highlight_max(color='lightgrey')
)


# ### How it works...

# ## Grouping by a Timestamp and another column

# ### How to do it...

# In[289]:


employee = pd.read_csv('data/employee.csv',
    parse_dates=['JOB_DATE', 'HIRE_DATE'],
    index_col='HIRE_DATE')
employee


# In[290]:


(employee
    .groupby('GENDER')
    ['BASE_SALARY']
    .mean()
    .round(-2)
)


# In[291]:


(employee
    .resample('10AS')
    ['BASE_SALARY']
    .mean()
    .round(-2)    
)


# In[292]:


(employee
   .groupby('GENDER')
   .resample('10AS')
   ['BASE_SALARY'] 
   .mean()
   .round(-2)
)


# In[293]:


(employee
   .groupby('GENDER')
   .resample('10AS')
   ['BASE_SALARY'] 
   .mean()
   .round(-2)
   .unstack('GENDER')
)


# In[294]:


employee[employee['GENDER'] == 'Male'].index.min()


# In[295]:


employee[employee['GENDER'] == 'Female'].index.min()


# In[296]:


(employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
)


# In[297]:


(employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
   .unstack('GENDER')
)


# ### How it works...

# ### There's more...

# In[298]:


sal_final = (employee
   .groupby(['GENDER', pd.Grouper(freq='10AS')]) 
   ['BASE_SALARY']
   .mean()
   .round(-2)
   .unstack('GENDER')
)
years = sal_final.index.year
years_right = years + 9
sal_final.index = years.astype(str) + '-' + years_right.astype(str)
sal_final


# In[299]:


cuts = pd.cut(employee.index.year, bins=5, precision=0)
cuts.categories.values


# In[300]:


(employee
    .groupby([cuts, 'GENDER'])
    ['BASE_SALARY'] 
    .mean()
    .unstack('GENDER')
    .round(-2)
)


# In[ ]:




