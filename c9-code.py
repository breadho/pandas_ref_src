#!/usr/bin/env python
# coding: utf-8

# # Grouping for Aggregation, Filtration, and Transformation

import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

for name in dir():
    if not name.startswith('_'):
        del globals()[name]

del(name)


# In[ ]:

import pandas as pd
import numpy as np
pd.set_option('max_columns', 10, 'max_rows', 10, 'max_colwidth', 50)


# ## Introduction

# ### Defining an Aggregation

# ### How to do it...

# In[ ]:


flights = pd.read_csv('data/flights.csv')
flights.head()


# In[ ]:


(flights
     .groupby('AIRLINE')
     .agg({'ARR_DELAY':'mean'})
)

test = flights.groupby('AIRLINE').agg({'ARR_DELAY':'mean'})

type(test)

# In[ ]:


(flights
     .groupby('AIRLINE')
     ['ARR_DELAY']
      .agg('mean')
)


# In[ ]:


(flights
    .groupby('AIRLINE')
    ['ARR_DELAY']
    .agg(np.mean)
)


# In[ ]:


(flights
    .groupby('AIRLINE')
    ['ARR_DELAY']
    .mean()
)


# ### How it works...

# In[ ]:


grouped = flights.groupby('AIRLINE')
type(grouped)


# ### There's more...

# In[ ]:


(flights
   .groupby('AIRLINE')
   ['ARR_DELAY']
   .agg(np.sqrt)
)


# ## Grouping and aggregating with multiple columns and functions

# ### How to do it...

# In[ ]:


(flights
    .groupby(['AIRLINE', 'WEEKDAY'])
    ['CANCELLED'] 
    .agg('sum')
)


# In[ ]:

# 하지만 책에서는 집계열의 쌍에 대해서도 리스트 사용을 권장
(flights
    .groupby(['AIRLINE', 'WEEKDAY']) 
    [['CANCELLED', 'DIVERTED']]
    .agg(['sum', 'mean'])
)


(flights
    .groupby(['AIRLINE', 'WEEKDAY']) 
    [['CANCELLED', 'DIVERTED']]
    .agg(['sum', 'mean'])
).equals((flights
    .groupby(['AIRLINE', 'WEEKDAY']) 
    ['CANCELLED', 'DIVERTED']
    .agg(['sum', 'mean'])
))


# In[ ]:


(flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['count', 'sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)

flights.CANCELLED.value_counts()
flights.DIVERTED.value_counts()

flights.columns





# In[ ]:


(flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg(sum_cancelled = pd.NamedAgg(column = 'CANCELLED', aggfunc = 'sum'),
         mean_cancelled = pd.NamedAgg(column = 'CANCELLED', aggfunc = 'mean'),
         size_cancelled = pd.NamedAgg(column = 'CANCELLED', aggfunc = 'size'),
         mean_air_time = pd.NamedAgg(column = 'AIR_TIME', aggfunc = 'mean'),
         var_air_time = pd.NamedAgg(column = 'AIR_TIME', aggfunc = 'var'))
)


# ### How it works...

# ### There's more...

# In[ ]:


res = (flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)

res.columns = ['_'.join(x) for x in
    res.columns.to_flat_index()]

type(res.columns.to_flat_index())

# In[ ]:


res


# In[ ]:


def flatten_cols(df):
    df.columns = ['_'.join(x) for x in
        df.columns.to_flat_index()]
    return df


# In[ ]:


res = (flights
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
    .pipe(flatten_cols)# .reindex 메서드는 펼치기를 지원하지 않으므로 
                       # .pipe 메서드를 활용(위에 정의한 flatten_cols 함수 이용)
)


# In[ ]:


res


# In[ ]:


res = (flights
    .assign(ORG_AIR=flights.ORG_AIR.astype('category')) 
     # 그룹화 열 중 하나가 범주형(category)이면 카티션곱(모든 조합) 발생
    .groupby(['ORG_AIR', 'DEST_AIR'])
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)
res

flights.ORG_AIR.value_counts()
flights.DEST_AIR.value_counts()

test = flights.assign(ORG_AIR=flights.ORG_AIR.astype('category'))
test.ORG_AIR.dtypes




# In[ ]:


res = (flights
    .assign(ORG_AIR=flights.ORG_AIR.astype('category'))
    .groupby(['ORG_AIR', 'DEST_AIR'], observed=True)
     # 모든 조합(카티션곱 폭발)을 방지하려면 observed = True 인자 사용 
    .agg({'CANCELLED':['sum', 'mean', 'size'],
          'AIR_TIME':['mean', 'var']})
)
res


# ## Removing the MultiIndex after grouping

# In[ ]:

    # 그룹화 후 다중 인덱스 제거

flights = pd.read_csv('data/flights.csv')
airline_info = (flights
    .groupby(['AIRLINE', 'WEEKDAY'])
    .agg({'DIST':['sum', 'mean'],
          'ARR_DELAY':['min', 'max']}) 
    .astype(int)
)
 
airline_info


# In[ ]:


airline_info.columns.get_level_values(0)
airline_info.columns.get_level_values(0).dtype

# In[ ]:


airline_info.columns.get_level_values(1)
airline_info.index.get_level_values(1)
airline_info.index.get_level_values(1)

# In[ ]:


airline_info.columns.to_flat_index()
airline_info.index.to_flat_index()

# In[ ]:


airline_info.columns = ['_'.join(x) for x in
    airline_info.columns.to_flat_index()]

# airline_info.index = ['_'.join(x) for x in
#     airline_info.index.to_flat_index()]


# In[ ]:


airline_info


# In[ ]:


airline_info.reset_index()


# In[ ]:


(flights
    .groupby(['AIRLINE', 'WEEKDAY'])
    .agg(dist_sum=pd.NamedAgg(column='DIST', aggfunc='sum'),
         dist_mean=pd.NamedAgg(column='DIST', aggfunc='mean'),
         arr_delay_min=pd.NamedAgg(column='ARR_DELAY', aggfunc='min'),
         arr_delay_max=pd.NamedAgg(column='ARR_DELAY', aggfunc='max'))
    .astype(int)
    .reset_index()
)

flights.AIRLINE.dtype
flights.WEEKDAY.dtype


# ### How it works...

# ### There's more...

# In[ ]:


(flights
    .groupby(['AIRLINE'] , as_index=False)
    ['DIST']
    .agg('mean')
    .round(0)
)


(flights
    .groupby(['AIRLINE'])
    ['DIST']
    .agg('mean')
    .round(0)
)

(flights
    .groupby(['AIRLINE'], sort = False)
    ['DIST']
    .agg('mean')
    .round(0)
)

(
 get_ipython()
 .run_line_magic('timeit',
                 'flights.groupby(["AIRLINE"])["DIST"].agg("mean").round(0)')
 )

# groupby 안에 sort 기능을 사용안하면 약간의 성능이 향상됨 
(get_ipython()
.run_line_magic('timeit', 
'(flights.groupby(["AIRLINE"], sort = False)["DIST"].agg("mean").round(0))')
)

# ## Grouping with a custom aggregation function

# ### How to do it...

# In[ ]:


college = pd.read_csv('data/college.csv')

(college
    .groupby('STABBR')
    ['UGDS']
    .agg(['mean', 'std'])
    .round(0)
)


# In[ ]:


def max_deviation(s):
    std_score = (s - s.mean()) / s.std()
    return std_score.abs().max()


# In[ ]:


(college
    .groupby('STABBR')
    ['UGDS']
    .agg(max_deviation)
    .round(1)
)

tmp = (college
    .groupby('STABBR')
    ['UGDS']
    .agg(max_deviation)
    .round(1)
)

tmp.max()
tmp.idxmax()
type(tmp[tmp.index == 'AS'])

tmp[tmp.index == tmp.idxmax()]
type(tmp[tmp.index == tmp.idxmax()])


# ### How it works...

# ### There's more...

# In[ ]:


(college
    .groupby('STABBR')
    ['UGDS', 'SATVRMID', 'SATMTMID']
    .agg(max_deviation)
    .round(1)
)


# In[ ]:


(college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATVRMID', 'SATMTMID'] 
    .agg([max_deviation, 'mean', 'std'])
    .round(1)
)


# In[ ]:


max_deviation.__name__


# In[ ]:


max_deviation.__name__ = 'Max Deviation'
tmp2 = (college
    .groupby(['STABBR', 'RELAFFIL']) 
    ['UGDS', 'SATVRMID', 'SATMTMID'] 
    .agg([max_deviation, 'mean', 'std'])
    .round(1)
)

tmp2.columns
tmp2.columns.rename({('UGDS', 'Max Deviation'),('UGDS', 'MD')}, inplace = True)



# ## Customizing aggregating functions with *args and **kwargs

# ### How to do it...

# In[ ]:

# 학부생 비율이 1,000에서 3,000 사이인 학교의 비율을 반환하는 함수 정의

def pct_between_1_3k(s):
    return (s
        .between(1_000, 3_000)
        .mean()
        * 100
    )


# In[ ]:
    
# 주와 종교에 대해 그룹화하고 비율을 계산함

(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg(pct_between_1_3k)
    .round(1)
)


# In[ ]: 
    
# 상한과 하한을 사용자가 지정할 수 있는 함수 작성
# 상한과 하한을 지정하여 그 비율을 산출하는 함수 작성

def pct_between(s, low, high):
    return s.between(low, high).mean() * 100


# In[ ]:

# 1,000 ~ 10,000으로 범위 지정

(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg(pct_between, 1_000, 10_000)
    .round(1)
)

# 명시적으로 키워드 매개변수를 사용해 동일한 결과를 얻을 수 있다.
(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg(pct_between, high = 10_000, low = 1_000)
    .round(1)
)


# ### How it works...

# ### There's more...

# In[ ]:

# 복수 집계함수를 호출하면서 일부 매개변수를 직접 제공하고 싶다면,
# 파이썬의 클로져 기능을 사용해 매개변수가 호출 환경에서 닫힌 상태로 되는
# 새로운 함수를 생성

def between_n_m(n, m):
    def wrapper(ser):
        return pct_between(ser, n, m)
    wrapper.__name__ = f'between_{n}_{m}'
    return wrapper


# In[ ]:


(college
    .groupby(['STABBR', 'RELAFFIL'])
    ['UGDS'] 
    .agg([between_n_m(1_000, 10_000), 'max', 'mean'])
    .round(1)
)


# ## Examining the groupby object

# ### How to do it...

# In[ ]: 


college = pd.read_csv('data/college.csv')
grouped = college.groupby(['STABBR', 'RELAFFIL'])
type(grouped)

college.RELAFFIL.value_counts()

# In[ ]:


print([attr for attr in dir(grouped) if not
    attr.startswith('_')])


# In[ ]:


grouped.ngroups


# In[ ]:


groups = list(grouped.groups)
groups[:6]


# In[ ]:

grouped.get_group(('FL', 1))




# In[ ]:


from IPython.display import display
for name, group in grouped:
    print(name)
    display(group.head(3))


# In[ ]:

# 그룹별로 잘려진 데이터들을 확인하는 방법

for name, group in grouped:
    print(name)
    print(group)
    break


# In[ ]:


grouped.head(2)


# ### How it works...

# ### There's more...

# In[ ]: 정수리스트가 제공될 때, 각 그룹에서 해당 행을 선택하는 .nth 메서드 사용

grouped.nth([1, -1]) # 각 그룹에서 첫 번째와 마지막 행을 선택함


# ## Filtering for states with a minority majority

# ### How to do it...

# In[ ]:


college = pd.read_csv('data/college.csv', index_col='INSTNM')
grouped = college.groupby('STABBR')
grouped.ngroups


# In[ ]:


college['STABBR'].nunique() # verifying the same number


# In[ ]:


def check_minority(df, threshold):
    minority_pct = 1 - df['UGDS_WHITE']
    total_minority = (df['UGDS'] * minority_pct).sum()
    total_ugds = df['UGDS'].sum()
    total_minority_pct = total_minority / total_ugds
    return total_minority_pct > threshold


# In[ ]:


college_filtered = grouped.filter(check_minority, threshold=.5)
college_filtered


# In[ ]:


college.shape


# In[ ]:


college_filtered.shape


# In[ ]:


college_filtered['STABBR'].nunique()


# ### How it works...

# ### There's more...

# In[ ]:


college_filtered_20 = grouped.filter(check_minority, threshold=.2)
college_filtered_20.shape


# In[ ]:


college_filtered_20['STABBR'].nunique()


# In[ ]:


college_filtered_70 = grouped.filter(check_minority, threshold=.7)
college_filtered_70.shape


# In[ ]:


college_filtered_70['STABBR'].nunique()


# ## Transforming through a weight loss bet

# ### How to do it...

# In[ ]:


weight_loss = pd.read_csv('data/weight_loss.csv')
weight_loss.query('Month == "Jan"')


# In[ ]:


def percent_loss(s):
    return ((s - s.iloc[0]) / s.iloc[0]) * 100


# In[ ]:


(weight_loss
    .query('Name=="Bob" and Month=="Jan"')
    ['Weight']
    .pipe(percent_loss)
)


# In[ ]:


(weight_loss
    .groupby(['Name', 'Month'])
    ['Weight'] 
    .transform(percent_loss)
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Name=="Bob" and Month in ["Jan", "Feb"]')
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
    .assign(winner=lambda df_:
            np.where(df_.Amy < df_.Bob, 'Amy', 'Bob'))
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
    .assign(winner=lambda df_:
            np.where(df_.Amy < df_.Bob, 'Amy', 'Bob'))
    .style.highlight_min(axis=1)
)


# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
    .assign(winner=lambda df_:
            np.where(df_.Amy < df_.Bob, 'Amy', 'Bob'))
    .winner
    .value_counts()
)


# ### How it works...

# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)))
    .query('Week == "Week 4"')
    .groupby(['Month', 'Name'])
    ['percent_loss']
    .first()
    .unstack()
)


# ### There's more...

# In[ ]:


(weight_loss
    .assign(percent_loss=(weight_loss
        .groupby(['Name', 'Month'])
        ['Weight'] 
        .transform(percent_loss)
        .round(1)),
            Month=pd.Categorical(weight_loss.Month,
                  categories=['Jan', 'Feb', 'Mar', 'Apr'],
                  ordered=True))
    .query('Week == "Week 4"')
    .pivot(index='Month', columns='Name',
           values='percent_loss')
)


# ## Calculating weighted mean SAT scores per state with apply

# ### How to do it...

# In[ ]:


college = pd.read_csv('data/college.csv')
subset = ['UGDS', 'SATMTMID', 'SATVRMID']
college2 = college.dropna(subset=subset)
college.shape


# In[ ]:


college2.shape


# In[ ]:


def weighted_math_average(df):
    weighted_math = df['UGDS'] * df['SATMTMID']
    return int(weighted_math.sum() / df['UGDS'].sum())


# In[ ]:

college2.UGDS
college2.SATMTMID

college2.groupby('STABBR').apply(weighted_math_average)


# In[ ]:


(college2
    .groupby('STABBR')
    .agg(weighted_math_average)
)


# In[ ]:


(college2
    .groupby('STABBR')
    ['SATMTMID'] 
    .agg(weighted_math_average)
)


# In[ ]:


def weighted_average(df):
   weight_m = df['UGDS'] * df['SATMTMID']
   weight_v = df['UGDS'] * df['SATVRMID']
   wm_avg = weight_m.sum() / df['UGDS'].sum()
   wv_avg = weight_v.sum() / df['UGDS'].sum()
   data = {'w_math_avg': wm_avg,
           'w_verbal_avg': wv_avg,
           'math_avg': df['SATMTMID'].mean(),
           'verbal_avg': df['SATVRMID'].mean(),
           'count': len(df)
   }
   return pd.Series(data)

(college2
    .groupby('STABBR')
    .apply(weighted_average)
    .astype(int)
)


# ### How it works...

# In[ ]:


(college
    .groupby('STABBR')
    .apply(weighted_average)
)


# ### There's more...

# In[ ]:


from scipy.stats import gmean, hmean

def calculate_means(df):
    df_means = pd.DataFrame(index=['Arithmetic', 'Weighted',
                                   'Geometric', 'Harmonic'])
    cols = ['SATMTMID', 'SATVRMID']
    for col in cols:
        arithmetic = df[col].mean()
        weighted = np.average(df[col], weights=df['UGDS'])
        geometric = gmean(df[col])
        harmonic = hmean(df[col])
        df_means[col] = [arithmetic, weighted,
                         geometric, harmonic]
    df_means['count'] = len(df)
    return df_means.astype(int)


(college2
    .groupby('STABBR')
    .apply(calculate_means)
)


# ## Grouping by continuous variables

# ### How to do it...

# In[ ]:


flights = pd.read_csv('data/flights.csv')
flights


# In[ ]:


bins = [-np.inf, 200, 500, 1000, 2000, np.inf]
cuts = pd.cut(flights['DIST'], bins=bins)
cuts


# In[ ]:


cuts.value_counts()
cuts.value_counts(normalize = True).round(2)*100

# In[ ]:


(flights
    .groupby(cuts)
    ['AIRLINE']
    .value_counts(normalize=True) 
    .round(3)
)


# ### How it works...

# ### There's more...

# In[ ]:


(flights
  .groupby(cuts)
  ['AIR_TIME']
  .quantile(q=[.25, .5, .75]) 
  .div(60)
  .round(2)
)


# In[ ]:


labels=['Under an Hour', '1 Hour', '1-2 Hours',
        '2-4 Hours', '4+ Hours']

cuts2 = pd.cut(flights['DIST'], bins=bins, labels=labels)

(flights
   .groupby(cuts2)
   ['AIRLINE']
   .value_counts(normalize=True) 
   .round(3) 
   .unstack() 
)


# ## Counting the total number of flights between cities

# ### How to do it...

# In[ ]:


flights = pd.read_csv('data/flights.csv')
flights_ct = flights.groupby(['ORG_AIR', 'DEST_AIR']).size()
flights_ct


# In[ ]:


flights_ct.loc[[('ATL', 'IAH'), ('IAH', 'ATL')]]


# In[ ]:


f_part3 = (flights  # doctest: +SKIP
  [['ORG_AIR', 'DEST_AIR']] 
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
)
f_part3


# In[ ]:


rename_dict = {0:'AIR1', 1:'AIR2'}  
(flights     # doctest: +SKIP
  [['ORG_AIR', 'DEST_AIR']]
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
  .rename(columns=rename_dict)
  .groupby(['AIR1', 'AIR2'])
  .size()
)


# In[ ]:


(flights     # doctest: +SKIP
  [['ORG_AIR', 'DEST_AIR']]
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
  .rename(columns=rename_dict)
  .groupby(['AIR1', 'AIR2'])
  .size()
  .loc[('ATL', 'IAH')]
)


# In[ ]:


(flights     # doctest: +SKIP
  [['ORG_AIR', 'DEST_AIR']]
  .apply(lambda ser:
         ser.sort_values().reset_index(drop=True),
         axis='columns')
  .rename(columns=rename_dict)
  .groupby(['AIR1', 'AIR2'])
  .size()
  .loc[('IAH', 'ATL')]
)


# ### How it works...

# ### There's more ...

# In[ ]:


data_sorted = np.sort(flights[['ORG_AIR', 'DEST_AIR']])
data_sorted[:10]


# In[ ]:


flights_sort2 = pd.DataFrame(data_sorted, columns=['AIR1', 'AIR2'])
flights_sort2.equals(f_part3.rename(columns={0:'AIR1',
    1:'AIR2'}))


# %%timeit
# flights_sort = (flights   # doctest: +SKIP
#     [['ORG_AIR', 'DEST_AIR']] 
#    .apply(lambda ser:
#          ser.sort_values().reset_index(drop=True),
#          axis='columns')
# )

# In[ ]:


get_ipython().run_cell_magic('timeit', 
                             '', 
                             "data_sorted = np.sort(flights[['ORG_AIR', 'DEST_AIR']])\nflights_sort2 = pd.DataFrame(data_sorted,\n    columns=['AIR1', 'AIR2'])")


# ## Finding the longest streak of on-time flights

# ### How to do it...

# In[ ]:


s = pd.Series([0, 1, 1, 0, 1, 1, 1, 0])
s


# In[ ]:

s1 = s.cumsum()
s1


# In[ ]:


s.mul(s1)


# In[ ]:


s.mul(s1).diff()


# In[ ]:


(s
    .mul(s.cumsum())
    .diff()
    .where(lambda x: x < 0)
)


# In[ ]:


(s
    .mul(s.cumsum())
    .diff()
    .where(lambda x: x < 0)
    .ffill()
)


# In[ ]:


(s
    .mul(s.cumsum())
    .diff()
    .where(lambda x: x < 0)
    .ffill()
    .add(s.cumsum(), fill_value=0)
)


# In[ ]:


flights = pd.read_csv('data/flights.csv')
(flights
    .assign(ON_TIME=flights['ARR_DELAY'].lt(15).astype(int))
    [['AIRLINE', 'ORG_AIR', 'ON_TIME']]
)


# In[ ]:


def max_streak(s):
    s1 = s.cumsum()
    return (s
       .mul(s1)
       .diff()
       .where(lambda x: x < 0) 
       .ffill()
       .add(s1, fill_value=0)
       .max()
    )


# In[ ]:


(flights
    .assign(ON_TIME=flights['ARR_DELAY'].lt(15).astype(int))
    .sort_values(['MONTH', 'DAY', 'SCHED_DEP']) 
    .groupby(['AIRLINE', 'ORG_AIR'])
    ['ON_TIME'] 
    .agg(['mean', 'size', max_streak])
    .round(2)
)


# ### How it works...

# ### There's more...

# In[ ]:


def max_delay_streak(df):
    df = df.reset_index(drop=True)
    late = 1 - df['ON_TIME']
    late_sum = late.cumsum()
    streak = (late
        .mul(late_sum)
        .diff()
        .where(lambda x: x < 0) 
        .ffill()
        .add(late_sum, fill_value=0)
    )
    last_idx = streak.idxmax()
    first_idx = last_idx - streak.max() + 1
    res = (df
        .loc[[first_idx, last_idx], ['MONTH', 'DAY']]
        .assign(streak=streak.max())
    )
    res.index = ['first', 'last']
    return res


# In[ ]:


(flights
    .assign(ON_TIME=flights['ARR_DELAY'].lt(15).astype(int))
    .sort_values(['MONTH', 'DAY', 'SCHED_DEP']) 
    .groupby(['AIRLINE', 'ORG_AIR']) 
    .apply(max_delay_streak) 
    .sort_values('streak', ascending=False)
)


# In[ ]:




