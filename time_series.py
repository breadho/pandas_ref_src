# -*- coding: utf-8 -*-
"""
Created on Tue Sep 21 15:40:37 2021

@author: begas05
"""


import os
os.getcwd()
os.chdir("D:/pandas_cookbook")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import datetime


# for name in dir():
#     if not name.startswith('_'):
#         del globals()[name]
        
# del(name)

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



'''
파이썬과 판다스 날짜 도구의 차이점 이해
'''

# 1. 먼저 datetime을 네임스페이스에 임포트하고 date, time, datetime 객체를 생성

import pandas as pd
import numpy as np
import datetime
date = datetime.date(year = 2013, month = 6, day = 7)
time = datetime.time(hour = 12, minute = 30, second = 19, microsecond = 463198)
dt = datetime.datetime(year = 2013, month = 6, day = 7,
                       hour = 12, minute = 30, second = 19,
                       microsecond = 463198)

print(f"date is {date}")
print(f"time is {time}")
print(f"datetime is {dt}")

# 2. 이제 datetim의 또 다른 주요 모듈인 timedelta 객체를 생성하고 출력해보자.
td = datetime.timedelta(weeks = 2, days = 5, hours = 10, 
                        minutes = 20, 
                        seconds = 6.73,
                        milliseconds = 99,
                        microseconds = 8)
td # 시간차

# 3. 이 td를 1단계의 date와 td 객체에 덧셈을 해보자.
print(f'new date is {date + td}')

print(f'new datetime is {dt + td}')

# 4. timedelta를 time 객체에 더하려 하면 오류가 발생함
print(f'new time is {time + td}') # 에러 발생


# 5. 이제 pandas로 돌아와 해당 Timestamp 객체를 살펴보자. 
'''
이 객체는 나노초 단위의 정밀도를 표현할 수 있는 어느 한 순간, 즉 특정 시각이다. 
Timestamp 생성자는 유연성이 뛰어나고 다양한 입력을 처리함
'''

pd.Timestamp(year = 2012, month = 12, day = 21, hour = 5,
             minute = 10, second = 8, microsecond = 99)

pd.Timestamp('2016/1/10')

pd.Timestamp('2014-5/10')

pd.Timestamp('Jan 3, 2019 20:45.56')

pd.Timestamp('2016-01-05T05:34:43.123456789')


# 6. 단일 정수나 부동소수점수를 Timestamp 생성자에 전달할 수도 있음.
'''
이는 유닉스 시간 (1970년 1월 1일)에 따라 경과된 나노초에 해당하는 날짜를 반환 
'''

pd.Timestamp(500)

pd.Timestamp(5000, unit = 'D')


# 7. Pandas에는 Timestamp 생성자와 유사하게 작동하는 to_datetime 함수가 제공됨
''' 이 함수는 DataFrame 문자열 열을 날짜로 변환하는데 유용함. '''

pd.to_datetime('2015-5-13')
pd.to_datetime('2015-13-5', dayfirst=True)

pd.to_datetime('Start Date: Sep 30, 2017 Start Time: 1:30 pm',
               format = 'Start Date: %b %d, %Y Start Time: %I:%M %p')

pd.to_datetime(100, unit = 'D', origin= '2013-1-1')



# 8. to_datetime 함수에는 더 많은 기능이 있다. 
# 이 함수는 전체 리스트나 문자열 Series 또는 정수를 Timestamp 객체로 변환할 수 있다.
# 단일 스칼라 값보다는 Series나 DataFrame과 상호작용할 가능성이 훨씬 높으므로 Timestamp보다 
# to_datetime을 사용하는 것이 훨씬 낫다.

s = pd.Series([10, 100, 1000, 10000])
pd.to_datetime(s, unit = 'D')

s = pd.Series(['12-5-2015', '14-1-2013', '20/12/2017', '40/23/2017'])
pd.to_datetime(s, dayfirst=True, errors = 'coerce') 
# 에러 항 강제 -> 오류 안나고 시간의 타입이 아닌 데이터는 NaT로 나타남 


# 9. pandas에는 시간량을 나타내는 Timedelta와 to_timedelta가 있다. 
'''
to_timedelta는 더 많은 기능을 가지고 있으며, 
전체 리스트나 Series를 Timedelta 객체로 변환할 수 있다.
'''

pd.Timedelta('12 days 5 hours 3 minutes 123456789 nanoseconds')

pd.Timedelta(days = 5, minutes = 7.34)

pd.Timedelta(100, unit = 'W')

pd.to_timedelta('67:15:45.454')

s = pd.Series([10, 100])
pd.to_timedelta(s, unit = 'h')

time_strings = ['2 days 24 minutes 89.67 seconds', '00:45:23.6']
pd.to_timedelta(time_strings)


# 10. Timedelta는 다른 Timestamp에 더하거나 뺄 수 있다. 
#     심지어 각각 나눠서 소수를 반환할 수 있다.

pd.Timedelta('12 days 5 hours 3 minutes')*2
(pd.Timestamp('1/1/2017') + pd.Timedelta('12 days 5 hours 3 minutes')*2)

td1 = pd.to_timedelta([10, 100], unit = 's')
td2 = pd.to_timedelta(['3 hours', '4 hours'])

td1 + td2

pd.Timedelta('12 days') / pd.Timedelta('3 days')



# 11. Timestamp와 Timedelta는 모두 속성과 메서드에 많은 특징을 가지고 있음

ts = pd.Timestamp('2016-10-1 4:23:23.9')

# 시간 올림
ts.ceil('h') 

ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second

ts.dayofweek, ts.dayofyear, ts.daysinmonth

ts.to_pydatetime()

td = pd.Timedelta(125.8723, unit = 'h')
td

td.round('min')

td.components

td.total_seconds()

'''

#### 시계열을 지능적으로 슬라이스 ####

- 부분 날짜 매칭을 이용해 DatetimeIndex를 가진 DataFrame을 선택하고 슬라이스함

'''

# 1. hdf5 파일 불러오기
crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes

# 2. REPORTED_DATE 열을 인덱스로 설정
crime = crime.set_index('REPORTED_DATE')
crime.index.sort_values

# 3. .loc 속성을 이용한 슬라이싱
crime.loc['2016-05-12 16:45:00']
len(crime.loc['2016-05-12 16:45:00'])

# 4. Timestamp를 인덱스로 설정해두면 
#    인덱스 값과 부분적으로 매칭되는 모든 행을 선택할 수 있음

crime.loc['2016-05-12']

# 5. 특정 날짜만이 아닌 전체 월, 연 또는 일 중 특정 시각도 선택할 수 있음
crime.loc['2016-05'].shape

crime.loc['2016'].shape

crime.loc['2016-05-12 03'].shape


# 6. 선택 문자열은 월 이름을 가질 수도 있다.
crime.loc['Dec 2015'].sort_index()


# 7. 월 이름이 포함된 다른 여러 문자열 형식도 작동함
crime.loc['2016 Sep, 15'].shape
crime.loc['21st October 2014 05'].shape


# 8. 선택 외에도 슬라이스 표기법을 사용하면 정확한 데이터 범위를 설정
crime.loc['2015-3-4':'2016-1-1'].sort_index()

# 9. 시간 폭 정밀하게 설정하기
crime.loc['2015-3-4 22':'2016-1-1 11:22:00'].sort_index()

crime.index[:2]


'''
시간 데이터로 열 필터링 
'''

# 1. hdf5 파일 crimes.h5에서 덴버 crimes 데이터셋을 읽은 후 열 형식을 살펴본다.

crime = pd.read_hdf('data/crime.h5', 'crime')
crime.dtypes

# 2. Boolean 인덱싱 사용 
(crime[crime.REPORTED_DATE == '2016-05-12 16:45:00'])

(crime[crime.REPORTED_DATE == '2016-05-12'])

(crime[crime.REPORTED_DATE.dt.date == '2016-05-12'])


# 4. 날짜의 일부가 매칭되는 것을 원하면 
#    부분 날짜 문자열을 지원하는 .between 메서드 사용
(crime[crime.REPORTED_DATE.between('2016-05-12', '2016-05-13')])

(crime[crime.REPORTED_DATE.between('2016-05', '2016-06')].shape)

(crime[crime.REPORTED_DATE.between('2016', '2017')].shape)

(crime[crime.REPORTED_DATE.between('2016-05-12 03', '2016-05-12 04')].shape)


# 6. 다른 문자열 패턴도 사용할 수 있다.
(crime[crime.REPORTED_DATE.between('2016 Sep, 15', '2016 Sep, 16')].shape)

(crime[crime.REPORTED_DATE.between('21st October 2014 05', '21st October 2014 06')].shape)


# 7. .loc는 폐구간으로, 시작과 끝 날짜를 모두 포함 -> .between 방식과 같음
#    하지만 포함하고자 하는 시간까지 모두 명시하는 것이 좋음

(crime[crime.REPORTED_DATE.between('2015-3-4', '2016-1-1 23:59:59')].shape)



#### 작동 원리 ####

'''
pandas 라이브러리는 인덱스 값을 슬라이스 할 수 있지만, 열을 슬라이스 할 수는 없음
열에서 슬라이스 하려면, .between 메서드를 사용
'''

lmask = crime.REPORTED_DATE >= '2015-3-4 22'
lmask.value_counts()

rmask = crime.REPORTED_DATE <= '2016-1-1 11:22:00'
rmask.value_counts()

crime[lmask & rmask].shape


# 인덱스에 대한 .loc와 열에 대한 .between 실행시간 측정

ctseries = crime.set_index('REPORTED_DATE')

%timeit ctseries.loc['2015-3-4': '2016-1-1']

%timeit crime[crime.REPORTED_DATE.between('2015-3-4', '2016-1-1')]


'''
DatetimeIndex에서만 작동하는 메서드 사용 
'''

# 1. crime hdf5 데이터셋을 읽은 다음, Reported_Date를 
#    인덱스로 설정해 DatetimeIndex가 되게 한다.

crime = (pd.read_hdf('data/crime.h5', 'crime').set_index('REPORTED_DATE'))

type(crime.index)


# 2. .between_time 메서드를 사용해 날짜와 상관없이 
#     오전 2시에서 오전 5시 사이에 발생한 모든 범죄를 선택함

crime.between_time('2:00', '5:00', include_end = False)

import datetime
crime.between_time(datetime.time(2, 0), datetime.time(5, 0), include_end = False)

# 같은 결과를 나타냄 


# 3. .at_time을 사용해 특정 시각의 모든 날짜를 선택함
crime.at_time('5:47')

# 4. .first 메서드는 시간에 대해 첫 n 세그먼트를 선택할 수 있는 방법
#   offset
crime_sort = crime.sort_index()
crime_sort.first(pd.offsets.MonthBegin(6))

crime_sort.first(pd.offsets.MonthEnd(6))

crime_sort.first(pd.offsets.MonthBegin(6, normalize = True))

crime_sort.loc[:'2012-06']


crime_sort.first('5D') # 5일간
crime_sort.first('5B') # 5영업일간

first_date = crime_sort.index[0]
first_date

first_date + pd.offsets.MonthBegin(6)
first_date + pd.offsets.MonthEnd(6)

'''
MonthBegin이나 MonthEnds 오프셋 모두 정확한 시간을 더하거나 빼지 않고 
일과 관계없이 다음 달의 시작이나 끝으로 효과적으로 올림함
내부적으로 .first 메서드는 DataFrame의 첫번째 인덱스 요소를 사용하고 
전달된 DateOffset을 더한다.
그런 다음 이 새로운 날짜까지 슬라이스 한다. 
'''

# 그런 다음 이 새로운 날짜까지 슬라이스한다. 예를 들어 4단계는 다음과 같다.

step4 = crime_sort.first(pd.offsets.MonthEnd(6))
end_dt = crime_sort.index[0] + pd.offsets.MonthEnd(6)
step4_internal = crime_sort[:end_dt]

step4.equals(step4_internal)



# 추가 사항
# 원하는 오프셋이 없을 때는 사용자 정의 DateOffset을 구축할 수도 있다.

dt = pd.Timestamp('2012-1-16 13:40')
dt + pd.DateOffset(months = 1)

do = pd.DateOffset(years = 2, months = 5, days = 3, hours = 8, seconds = 10)
pd.Timestamp('2012-1-22 03:22') + do



'''
주간 범죄 수 계산
'''

# 1. crime hdf5 데이터 셋을 읽고 REPORTED_DATE를 인덱스로 설정한다.
#    그런 다음 정렬해 나머지 예제의 성능을 향상시킨다. 

crime_sort = (pd.read_hdf('data/crime.h5', 'crime')
              .set_index('REPORTED_DATE')
              .sort_index()
              )

# 2. 주당 범죄수를 세려면 각 주별로 그룹을 구성해야 함

crime_sort.resample('W')

(crime_sort.resample('W').size())

len(crime_sort.loc[:'2012-1-8'])
len(crime_sort.loc['2012-1-9':'2012-1-15'])

# 5. 주의 마지막 날을 일요일이 아닌 다른 요일로 선택

(crime_sort
 .resample('W-THU')
 .size()
)


# 6. .resample의 거의 모든 기능은 .groupby 메서드로 재현할 수 있다. 
#     유일한 차이점은 오프셋을 pd.Grouper 객체에 전달해야 한다는 것

weekly_crimes = (crime_sort.groupby(pd.Grouper(freq = 'W')).size())

weekly_crimes 

# .resample과 함께 쓸 수 있는 객체 

r = crime_sort.resample('W')
[attr for attr in dir(r) if attr[0].islower()]

dir(r)


# 추가 사항
'''
인덱스에 Timestamp가 포함돼 있지 않아도 .resample을 사용할 수 있다.
'''
crime = pd.read_hdf('data/crime.h5', 'crime')
weekly_crimes2 = crime.resample('W', on = 'REPORTED_DATE').size()
weekly_crimes2.equals(weekly_crimes)

weekly_crimes_gby2 = (crime.groupby(pd.Grouper(key = 'REPORTED_DATE', freq = 'W'))
                      .size()
                      )

weekly_crimes2.equals(weekly_crimes)

import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (16, 4))
weekly_crimes.plot(title = 'All Denver Crimes', ax = ax)


'''
주간 범죄와 교통사고를 별도로 집계
.resample 메서드를 사용해 분기별로 그룹화한 다음 범죄와 교통사고 건수를 개별적으로 합산
'''

# 작동방법

# 1. crime hdf5 데이터 셋을 읽고 REPORTED_DATE를 인덱스로 설정
# 그런 다음 정렬을 실시하여, 성능을 향상시킴 
crime = (pd.read_hdf('data/crime.h5', 'crime')
         .set_index('REPORTED_DATE')
         .sort_index()
         ) 

# 2. .resample 메서드를 사용해 분기별로 그룹화하고 
#    각 그룹에서 IS_CRIME과 IS_TRAFFIC 열을 합산함

(crime.resample('Q')
       [['IS_CRIME', 'IS_TRAFFIC']]
       .sum()
)

# 3. 모두 분기의 마지막 날이 나타난 점에 주목하자.
#    이는 오프셋 별칭을 Q로 사용했기 때문
#    QS로 설정해 분기 시작으로 나타내보자.

(crime.resample('QS')
       [['IS_CRIME', 'IS_TRAFFIC']]
       .sum()
)

# 4. 두 번째 분기의 데이터가 올바른지 확인해 결과를 검증
(crime.loc['2012-4-1':'2012-6-30', ['IS_CRIME', 'IS_TRAFFIC']].sum())

# 5. 이 연산은 .groupby 메서드를 사용해 복제
(crime.groupby(pd.Grouper(freq = 'QS'))
       [['IS_CRIME', 'IS_TRAFFIC']]
       .sum())

# 6. 시간에 따른 범죄와 교통사고의 경향을 도식화해 시각화 해보기

fig, ax = plt.subplots(figsize = (16, 4))
(crime.groupby(pd.Grouper(freq = 'Q'))
       [['IS_CRIME', 'IS_TRAFFIC']]
       .sum()
       .plot(color = ['black', 'blue'], 
             ax = ax,
             title = 'Denver Crimes and Traffic Accidents'))


# 번외: 분기의 시작이 3월 1일에 시작되게 하려면 QS-MAR을 사용 
(crime.resample('QS-MAR')[['IS_CRIME', 'IS_TRAFFIC']].sum())



# 추가 사항
''' 다른 시각적 관점을 보고자 건수 대신 범죄와 교통사고의 백분율 증가를 그려볼 수 있음 '''

# 모든 데이터를 첫 번째 행으로 나누고 다시 그려보자 
crime_begin = (crime.resample('Q')[['IS_CRIME', 'IS_TRAFFIC']]
                    .sum()
                    .iloc[0])

fig, ax = plt.subplots(figsize = (12, 3))
(crime.resample('Q')[['IS_CRIME', 'IS_TRAFFIC']].sum()
      .div(crime_begin)
      .sub(1)
      .round(2)
      .mul(100)
      .plot.bar(color = ['skyblue', 'pink'],
                ax = ax,
                title = 'Denver Crimes and Traffic Accidents % Increase')
      )


'''
주별, 연도별 범죄 측정
'''

# 주별, 연도별 범죄를 동시에 측정하려면 Timestamp에서 정보를 추출할 수 있는 기능이 필요
# 고맙게도 이 기능은 모든 Timestamp 열에 .dt라는 속성으로 구축되어 있음 

# 이 예제에서는 .dt 속성을 사용해 각 범죄 발생 요일과 연도를 series로 제공
# 이 series를 모두 사용해 그룹을 구성, 모든 범죄를 계산함
# 범죄 총계에 대한 히트맵을 만들기 전에 부분 연도와 인구를 고려하도록 데이터를 조정

# 작동 방법
# 1. 덴버 crime hdf5 데이터셋을 읽어 들이되 REPORTED_DATE 열을 그대로 유지
crime = pd.read_hdf('data/crime.h5', 'crime')
crime

# 2. 모든 Timestamp 열은 특수 속성인 .dt를 갖고 있으며 이를 통해 날짜에만 특화된
#    다양한 추가 속성과 메서드에 접근할 수 있음
#    각 REPORTED_DATE의 요일 이름을 찾은 후 이 값을 세어보자.

(crime['REPORTED_DATE'].dt.day_name().value_counts())

crime['REPORTED_DATE'].dt


'''
r = crime['REPORTED_DATE'].dt
[attr for attr in dir(r) if attr[0].islower()]
'''

# 3. 주말에는 현저하게 사건이나 교통사고가 줄어드는 것을 보임
#    이 데이터를 정확한 요일별로 정렬하고 수평 막대 그리기

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
title = 'Denver Crimes and Traffic Accidents per Weekday'

fig, ax = plt.subplots(figsize = (6, 4))
(crime['REPORTED_DATE'].dt.day_name()
                         .value_counts()
                         .reindex(days)
                         .plot.barh(title = title, ax = ax))

# 4. 연도별 범죄 건수에 대해서도 비슷한 절차를 실행 가능 
title = 'Denver Crimes and Traffic Accidents per Year'
fig, ax = plt.subplots(figsize = (6, 4))
(crime['REPORTED_DATE'].dt.year.value_counts()
                         .sort_index()
                         .plot.barh(title = title, ax = ax))

# 5. 주별, 연도별 모두 그룹화해야 하는데, 
#    방법 중 하나는 이 속성을 .groupby 메서드에서 사용하는 것
(crime.groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
                crime['REPORTED_DATE'].dt.day_name().rename('day')]).size())

# 6. 더 가독성 있는 테이블을 얻으려면 .unstack 메서드를 사용
(crime.groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
                crime['REPORTED_DATE'].dt.day_name().rename('day')])
                .size()
                .unstack('day'))


# 7. 공정한 비교를 위해 2017년의 마지막 데이터의 날을 살펴봄
criteria = crime['REPORTED_DATE'].dt.year == 2017
crime.loc[criteria, 'REPORTED_DATE'].dt.dayofyear.max()

# 8. 초보적 접근법은 연중 일정한 범죄율을 가정하고 2017년의 모든 값에 
#    365/272를 곱하는 것이다.
#    그러나 여기서는 그보다 좀 더 나은 방법을 써서 과거 데이터를 보고
#    연중 처음 272일 동안 발생한 범죄의 평균 비율을 계산함
 
round(272/365, 3)

crime_pct = (crime['REPORTED_DATE'].dt.dayofyear.le(272)
                                     .groupby(crime.REPORTED_DATE.dt.year)
                                     .mean().mul(100).round(2))
crime_pct


crime_pct.loc[2012:2016].median()


# 8. 우연히도 처음 272일 동안 발생한 범죄의 비율은 해당 연도의 일 비율과 거의 비례
#    2017년도의 행을 갱신하고 요일 순서와 일치하게 열 순서를 변경

def update_2017(df_):
    df_.loc[2017] = (df_.loc[2017].div(.748).astype('int'))
    return df_


(crime.groupby([crime['REPORTED_DATE'].
                dt.year.rename('year'),
           crime['REPORTED_DATE'].
           dt.day_name()
           .rename('day')]).size().unstack('day').loc[2017].div(.748).reindex(columns = days))


temp = crime.groupby([crime['REPORTED_DATE'].
                dt.year.rename('year'),
           crime['REPORTED_DATE'].
           dt.day_name()
           .rename('day')]).size().unstack('day').pipe(update_2017)


# 10. 막대나 선 그래프를 그릴 수도 있지만 seaborn 라이브러리에 있는 히트맵을
#     사용하기에 적절한 상황이다. 


table = (crime.groupby([crime['REPORTED_DATE'].dt.year.rename('year'), 
                        crime['REPORTED_DATE'].dt.day_name().rename('day')])
         .size()
         .unstack('day')
         .pipe(update_2017)
         .reindex(columns = days)
         )


import seaborn as sns

fig, ax = plt.subplots(figsize = (6, 6))
sns.heatmap(table, cmap = 'viridis_r', ax = ax)


# 11. 범죄가 매년 증가하는 것으로 보이지만 이 데이터는 증가하는 인구수를 감안하지 못함
denver_pop = pd.read_csv('data/denver_pop.csv', index_col = 'Year')
denver_pop


# 12. 많은 범죄 척도는 거주자 10만 명당 범죄 건수로 보고됨
#     이제 인구를 10만명으로 나눈 다음 이 숫자로 범죄 건수를 나눠 
#     거주자 10만명당 범죄율을 계산
# 

den_100k = denver_pop.div(100_000).squeeze()

normalized = (crime.groupby([crime['REPORTED_DATE'].dt.year.rename('year'),
                             crime['REPORTED_DATE'].dt.day_name().rename('day')])
              .size()
              .unstack('day')
              .pipe(update_2017)
              .reindex(columns = days)
              .div(den_100k, axis = 'index')
              .astype(int)
              )

normalized

# 13. 이번에도 히트맵 이용 그래프 그리기 

import seaborn as sns
fig, ax = plt.subplots(figsize = (6, 4))
sns.heatmap(normalized, cmap = 'viridis_r', ax = ax)

fig, ax = plt.subplots(figsize = (6, 4))
sns.heatmap(normalized, cmap = 'YlGnBu', ax = ax)


# 작동 원리
(crime['REPORTED_DATE'].dt.day_name().value_counts().loc[days])

# 연도와 요일별로 비교하기 어려운 테이블을 가독성을 높이고자 변환
# 교차테이블 작성
(crime.assign(year = crime.REPORTED_DATE.dt.year,
              day = crime.REPORTED_DATE.dt.day_name())
        .pipe(lambda df_: pd.crosstab(df_.year, df_.day)) 
 )


'''
timeIndex를 사용해 익명 함수로 그룹화
'''
# DatetimeIndex와 함께 DataFrame을 사용하면 이 장의 여러 예제에서 볼 수 있듯이 
# 새롭고 다양한 작업을 수행할 수 있다. 

# 이 예제에서는 DatetimeIndex가 있는 DataFrame에
# .groupby 메서드를 사용하는 다양한 방법을 보여줌

# 1. 덴버 crime hdf5 파일을 읽고 REPORTED_DATE 열을 인덱스에 넣은 다음 정렬함
crime = (pd.read_hdf('data/crime.h5', 'crime')
         .set_index('REPORTED_DATE')
         .sort_index())

# 2. DatetimeIndex는 pandas Timestamp와 동일한 여러 속성과 메서드를 가지고 있음
common_attrs = (set(dir(crime.index)) & set(dir(pd.Timestamp)))
[attr for attr in common_attrs if attr[0] != '_']


# 3. .index를 사용하면 요일 이름을 알 수 있음
crime.index.day_name().value_counts()

# 4. .groupby 메서드는 함수를 인수로 취할 수 있음. 
#    이 함수에는 .index가 전달되고 반환값은 그룹을 형성하는 데 사용됨
# .index를 요일 이름으로 바꾸고 범죄와 교통사고 건수를 별도로 계산하는 함수로 그룹화

(crime.groupby(lambda idx: idx.day_name())
       [['IS_CRIME', 'IS_TRAFFIC']].sum())

# 5. 함수 리스트를 사용하면 일중 시간과 연도 모두에 대해 그룹화 할 수 있음
funcs = [lambda idx: idx.round('2h').hour, lambda idx: idx.year]
(crime.groupby(funcs)[['IS_CRIME', 'IS_TRAFFIC']].sum().unstack())


'''
Timestamp와 다른 열을 기준으로 그룹화
'''

# 작동 방법 
# 1. employee 데이터셋을 읽고 HIRE_DATE 열에 DatetimeIndex를 생성함
employee = pd.read_csv('data/employee.csv',
                       parse_dates = ['JOB_DATE', 'HIRE_DATE'],
                       index_col = 'HIRE_DATE')
employee

# 2. 먼저 성별에 대해 그룹화하고 각가의 평균 급여를 살펴봄
(employee.groupby('GENDER')['BASE_SALARY'].mean().round(-2))

# 3. 고용일에 따른 평균 급여를 살펴보고 10년 단위 버킷으로 그룹화함
(employee.resample('10AS')['BASE_SALARY'].mean().round(-2))

# 4. 성별과 10년 구간을 둘 다 그룹화하려 했다면, .groupby 호출 후에 바로 .resample을 호출함
(employee.groupby('GENDER').resample('10AS')['BASE_SALARY'].mean().round(-2))


(employee.groupby('GENDER').resample('10AS')['BASE_SALARY']
 .mean()
 .round(-2)
 .unstack('GENDER'))


# 6. 남성과 여성이 10년 기간이 같은 날짜에서 시작되지 않는다. 
# 이 문제는 데이터가 처음에 성별로 그룹화된 후 각 성별 내에서 채용 날짜를 기준으로
# 더 많은 그룹을 형성했기 때문에 발생함
# 첫 번째 고용된 남성은 1958년이고 첫 번째 고용된 여성은 1975년이 맞는지 확인해 본다.

employee[employee['GENDER'] == 'Male'].index.min()

employee[employee['GENDER'] == 'Female'].index.min()


# 7. 이 문제를 해결하려면 날짜를 성별과 함께 그룹화 해야함
#    이는 오직 .groupby 메서드로만 가능함
(employee.groupby(['GENDER', pd.Grouper(freq = '10AS')])
     ['BASE_SALARY']
     .mean()
     .round(-2))


# 8. 이제 성별을 .unstack해 행들이 완벽히 정렬되게 함
(employee.groupby(['GENDER', pd.Grouper(freq = '10AS')])
     ['BASE_SALARY']
     .mean()
     .round(-2).unstack('GENDER'))



# 9. 외부의 시각에서는 8단계의 출력 행이 
#    10년 간격을 의미한다는 것이 분명하지 않을 수 있음
#    인덱스 레이블을 개선하는 한 가지 방법은 각 시간 간격의 시작과 끝을 표시하는 것
#    현재 인덱스 연도에 9를 더하면 이렇게 할 수 있음

sal_final = (employee.groupby(['GENDER', pd.Grouper(freq = '10AS')])
             ['BASE_SALARY']
             .mean()
             .round(-2)
             .unstack('GENDER')
             )

years = sal_final.index.year
years_right = years + 9 

sal_final.index = years.astype(str) + '-' + years_right.astype(str)
sal_final



# 9-1. cut 함수 이용한 방법 수행 
cuts = pd.cut(employee.index.year, bins = 5, precision = 0)
cuts.categories.values
(employee.groupby([cuts, 'GENDER'])['BASE_SALARY'].mean().unstack('GENDER').round(-2))








