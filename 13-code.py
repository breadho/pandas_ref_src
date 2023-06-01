#!/usr/bin/env python
# coding: utf-8

# # Visualization with Matplotlib, Pandas, and Seaborn

# ## Introduction

# ## Getting started with matplotlib

# In[20]:


get_ipython().run_line_magic('matplotlib', 'inline')


# ## Object-oriented guide to matplotlib

# In[21]:


import matplotlib.pyplot as plt
x = [-3, 5, 7]
y = [10, 2, 5]
fig = plt.figure(figsize=(15,3))
plt.plot(x, y)
plt.xlim(0, 10)
plt.ylim(-3, 8)
plt.xlabel('X Axis')
plt.ylabel('Y axis')
plt.title('Line Plot')
plt.suptitle('Figure Title', size=20, y=1.03)
fig.savefig('c13-fig1.png', dpi=300, bbox_inches='tight')


# In[22]:


from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from IPython.core.display import display
fig = Figure(figsize=(15, 3))
FigureCanvas(fig)  
ax = fig.add_subplot(111)
ax.plot(x, y)
ax.set_xlim(0, 10)
ax.set_ylim(-3, 8)
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_title('Line Plot')
fig.suptitle('Figure Title', size=20, y=1.03)
display(fig)
fig.savefig('c13-fig2.png', dpi=300, bbox_inches='tight')


# In[23]:


fig, ax = plt.subplots(figsize=(15,3))
ax.plot(x, y)
ax.set(xlim=(0, 10), ylim=(-3, 8),
    xlabel='X axis', ylabel='Y axis',
    title='Line Plot')
fig.suptitle('Figure Title', size=20, y=1.03)
fig.savefig('c13-fig3.png', dpi=300, bbox_inches='tight')


# ### How to do it...

# In[24]:


import matplotlib.pyplot as plt


# In[25]:


fig, ax = plt.subplots(nrows=1, ncols=1)
fig.savefig('c13-step2.png', dpi=300)         


# In[26]:


type(fig)


# In[27]:


type(ax)


# In[28]:


fig.get_size_inches()


# In[29]:


fig.set_size_inches(14, 4)
fig.savefig('c13-step4.png', dpi=300)         
fig


# In[30]:


fig.axes


# In[31]:


fig.axes[0] is ax


# In[32]:


fig.set_facecolor('.7')
ax.set_facecolor('.5')
fig.savefig('c13-step7.png', dpi=300, facecolor='.7')  
fig


# In[33]:


ax_children = ax.get_children()
ax_children


# In[34]:


spines = ax.spines
spines


# In[35]:


spine_left = spines['left']
spine_left.set_position(('outward', -100))
spine_left.set_linewidth(5)
spine_bottom = spines['bottom']
spine_bottom.set_visible(False)
fig.savefig('c13-step10.png', dpi=300, facecolor='.7')
fig


# In[36]:


ax.xaxis.grid(True, which='major', linewidth=2,
    color='black', linestyle='--')
ax.xaxis.set_ticks([.2, .4, .55, .93])
ax.xaxis.set_label_text('X Axis', family='Verdana',
    fontsize=15)
ax.set_ylabel('Y Axis', family='Gotham', fontsize=20)
ax.set_yticks([.1, .9])
ax.set_yticklabels(['point 1', 'point 9'], rotation=45)
fig.savefig('c13-step11.png', dpi=300, facecolor='.7')         


# ### How it works...

# In[37]:


plot_objects = plt.subplots(nrows=1, ncols=1)
type(plot_objects)


# In[38]:


fig = plot_objects[0]
ax = plot_objects[1]
fig.savefig('c13-1-works1.png', dpi=300)         


# In[39]:


fig, axs = plt.subplots(2, 4)
fig.savefig('c13-1-works2.png', dpi=300)         


# In[40]:


axs


# In[41]:


ax = axs[0][0]
fig.axes == fig.get_axes()


# In[42]:


ax.xaxis == ax.get_xaxis()


# In[43]:


ax.yaxis == ax.get_yaxis()


# ### There's more...

# In[44]:


ax.xaxis.properties()


# ## Visualizing data with matplotlib

# ### How to do it...

# In[45]:


import pandas as pd
import numpy as np
alta = pd.read_csv('data/alta-noaa-1980-2019.csv')
alta


# In[46]:


data = (alta
    .assign(DATE=pd.to_datetime(alta.DATE))
    .set_index('DATE')
    .loc['2018-09':'2019-08']
    .SNWD
)
data


# In[47]:


blue = '#99ddee'
white = '#ffffff'
fig, ax = plt.subplots(figsize=(12,4), 
     linewidth=5, facecolor=blue)
ax.set_facecolor(blue)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='x', colors=white)
ax.tick_params(axis='y', colors=white)
ax.set_ylabel('Snow Depth (in)', color=white)
ax.set_title('2009-2010', color=white, fontweight='bold')
ax.fill_between(data.index, data, color=white)
fig.savefig('c13-alta1.png', dpi=300, facecolor=blue)  


# In[48]:


import matplotlib.dates as mdt
blue = '#99ddee'
white = '#ffffff'


# In[49]:


def plot_year(ax, data, years):
    ax.set_facecolor(blue)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='x', colors=white)
    ax.tick_params(axis='y', colors=white)
    ax.set_ylabel('Snow Depth (in)', color=white)
    ax.set_title(years, color=white, fontweight='bold')
    ax.fill_between(data.index, data, color=white)


# In[50]:


years = range(2009, 2019)
fig, axs = plt.subplots(ncols=2, nrows=int(len(years)/2), 
    figsize=(16, 10), linewidth=5, facecolor=blue)
axs = axs.flatten()
max_val = None
max_data = None
max_ax = None
for i,y in enumerate(years):
    ax = axs[i]
    data = (alta
       .assign(DATE=pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc[f'{y}-09':f'{y+1}-08']
       .SNWD
    )
    if max_val is None or max_val < data.max():
        max_val = data.max()
        max_data = data
        max_ax = ax
    ax.set_ylim(0, 180)
    years = f'{y}-{y+1}'
    plot_year(ax, data, years)
max_ax.annotate(f'Max Snow {max_val}', 
   xy=(mdt.date2num(max_data.idxmax()), max_val), 
   color=white)


# In[51]:


fig.suptitle('Alta Snowfall', color=white, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('c13-alta2.png', dpi=300, facecolor=blue)  


# ### How it works...

# ### There's more...

# In[121]:


years = range(2009, 2019)
fig, axs = plt.subplots(ncols=2, nrows=int(len(years)/2), 
    figsize=(16, 10), linewidth=5, facecolor=blue)
axs = axs.flatten()
max_val = None
max_data = None
max_ax = None
for i,y in enumerate(years):
    ax = axs[i]
    data = (alta.assign(DATE=pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc[f'{y}-09':f'{y+1}-08']
       .SNWD
       .interpolate()
    )
    if max_val is None or max_val < data.max():
        max_val = data.max()
        max_data = data
        max_ax = ax
    ax.set_ylim(0, 180)
    years = f'{y}-{y+1}'
    plot_year(ax, data, years)
max_ax.annotate(f'Max Snow {max_val}', 
   xy=(mdt.date2num(max_data.idxmax()), max_val), 
   color=white)
plt.tight_layout()


# In[53]:


fig.suptitle('Alta Snowfall', color=white, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('c13-alta3.png', dpi=300, facecolor=blue)  


# In[54]:


(alta
    .assign(DATE=pd.to_datetime(alta.DATE))
    .set_index('DATE')
    .SNWD
    .to_frame()
    .assign(next=lambda df_:df_.SNWD.shift(-1),
            snwd_diff=lambda df_:df_.next-df_.SNWD)
    .pipe(lambda df_: df_[df_.snwd_diff.abs() > 50])
)


# In[55]:


def fix_gaps(ser, threshold=50):
    'Replace values where the shift is > threshold with nan'
    mask = (ser
       .to_frame()
       .assign(next=lambda df_:df_.SNWD.shift(-1),
               snwd_diff=lambda df_:df_.next-df_.SNWD)
       .pipe(lambda df_: df_.snwd_diff.abs() > threshold)
    )
    return ser.where(~mask, np.nan)


# In[56]:


years = range(2009, 2019)
fig, axs = plt.subplots(ncols=2, nrows=int(len(years)/2), 
    figsize=(16, 10), linewidth=5, facecolor=blue)
axs = axs.flatten()
max_val = None
max_data = None
max_ax = None
for i,y in enumerate(years):
    ax = axs[i]
    data = (alta.assign(DATE=pd.to_datetime(alta.DATE))
       .set_index('DATE')
       .loc[f'{y}-09':f'{y+1}-08']
       .SNWD
       .pipe(fix_gaps)
       .interpolate()
    )
    if max_val is None or max_val < data.max():
        max_val = data.max()
        max_data = data
        max_ax = ax
    ax.set_ylim(0, 180)
    years = f'{y}-{y+1}'
    plot_year(ax, data, years)
max_ax.annotate(f'Max Snow {max_val}', 
   xy=(mdt.date2num(max_data.idxmax()), max_val), 
   color=white)


# In[57]:


fig.suptitle('Alta Snowfall', color=white, fontweight='bold')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.savefig('c13-alta4.png', dpi=300, facecolor=blue)  


# ## Plotting basics with pandas

# ### How to do it..

# In[58]:


df = pd.DataFrame(index=['Atiya', 'Abbas', 'Cornelia',
    'Stephanie', 'Monte'],
    data={'Apples':[20, 10, 40, 20, 50],
          'Oranges':[35, 40, 25, 19, 33]})


# In[59]:


df


# In[60]:


color = ['.2', '.7']
ax = df.plot.bar(color=color, figsize=(16,4))
ax.get_figure().savefig('c13-pdemo-bar1.png', dpi=300, bbox_inches='tight')


# In[61]:


ax = df.plot.kde(color=color, figsize=(16,4))
ax.get_figure().savefig('c13-pdemo-kde1.png')


# In[62]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
fig.suptitle('Two Variable Plots', size=20, y=1.02)
df.plot.line(ax=ax1, title='Line plot')
df.plot.scatter(x='Apples', y='Oranges', 
    ax=ax2, title='Scatterplot')
df.plot.bar(color=color, ax=ax3, title='Bar plot')
fig.savefig('c13-pdemo-scat.png', dpi=300, bbox_inches='tight')


# In[63]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
fig.suptitle('One Variable Plots', size=20, y=1.02)
df.plot.kde(color=color, ax=ax1, title='KDE plot')
df.plot.box(ax=ax2, title='Boxplot')
df.plot.hist(color=color, ax=ax3, title='Histogram')
fig.savefig('c13-pdemo-kde2.png', dpi=300, bbox_inches='tight')


# ### How it works...

# ### There's more...

# In[64]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16,4))
df.sort_values('Apples').plot.line(x='Apples', y='Oranges',
      ax=ax1)
df.plot.bar(x='Apples', y='Oranges', ax=ax2)
df.plot.kde(x='Apples', ax=ax3)
fig.savefig('c13-pdemo-kde3.png', dpi=300, bbox_inches='tight')


# ## Visualizing the flights dataset

# ### How to do it...

# In[65]:


flights = pd.read_csv('data/flights.csv')
flights


# In[66]:


cols = ['DIVERTED', 'CANCELLED', 'DELAYED']
(flights
    .assign(DELAYED=flights['ARR_DELAY'].ge(15).astype(int),
            ON_TIME=lambda df_:1 - df_[cols].any(axis=1))
    .select_dtypes(int)
    .sum()
)


# In[67]:


fig, ax_array = plt.subplots(2, 3, figsize=(18,8))
(ax1, ax2, ax3), (ax4, ax5, ax6) = ax_array
fig.suptitle('2015 US Flights - Univariate Summary', size=20)
ac = flights['AIRLINE'].value_counts()
ac.plot.barh(ax=ax1, title='Airline')
(flights
    ['ORG_AIR']
    .value_counts()
    .plot.bar(ax=ax2, rot=0, title='Origin City')
)
(flights
    ['DEST_AIR']
    .value_counts()
    .head(10)
    .plot.bar(ax=ax3, rot=0, title='Destination City')
)
(flights
    .assign(DELAYED=flights['ARR_DELAY'].ge(15).astype(int),
            ON_TIME=lambda df_:1 - df_[cols].any(axis=1))
    [['DIVERTED', 'CANCELLED', 'DELAYED', 'ON_TIME']]
    .sum()
    .plot.bar(ax=ax4, rot=0,
         log=True, title='Flight Status')
)
flights['DIST'].plot.kde(ax=ax5, xlim=(0, 3000),
    title='Distance KDE')
flights['ARR_DELAY'].plot.hist(ax=ax6,
    title='Arrival Delay',
    range=(0,200)
)
fig.savefig('c13-uni1.png')


# In[68]:


df_date = (flights
    [['MONTH', 'DAY']]
    .assign(YEAR=2015,
            HOUR=flights['SCHED_DEP'] // 100,
            MINUTE=flights['SCHED_DEP'] % 100)
)
df_date


# In[69]:


flight_dep = pd.to_datetime(df_date)
flight_dep


# In[70]:


flights.index = flight_dep
fc = flights.resample('W').size()
fc.plot.line(figsize=(12,6), title='Flights per Week', grid=True)
fig.savefig('c13-ts1.png', dpi=300, bbox_inches='tight')


# In[71]:


def interp_lt_n(df_, n=600):
    return (df_
        .where(df_ > n)
        .interpolate(limit_direction='both')
)
fig, ax = plt.subplots(figsize=(16,4))
data = (flights
    .resample('W')
    .size()
)
(data
    .pipe(interp_lt_n)
    .iloc[1:-1]
    .plot.line(color='black', ax=ax)
)
mask = data<600
(data
     .pipe(interp_lt_n)
     [mask]
     .plot.line(color='.8', linewidth=10)
) 
ax.annotate(xy=(.8, .55), xytext=(.8, .77),
            xycoords='axes fraction', s='missing data',
            ha='center', size=20, arrowprops=dict())
ax.set_title('Flights per Week (Interpolated Missing Data)')
fig.savefig('c13-ts2.png')


# In[72]:


fig, ax = plt.subplots(figsize=(16,4))
(flights
    .groupby('DEST_AIR')
    ['DIST'] 
    .agg(['mean', 'count']) 
    .query('count > 100') 
    .sort_values('mean') 
    .tail(10) 
    .plot.bar(y='mean', rot=0, legend=False, ax=ax,
        title='Average Distance per Destination')
)
fig.savefig('c13-bar1.png')


# In[73]:


fig, ax = plt.subplots(figsize=(8,6))
(flights
    .reset_index(drop=True)
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .plot.scatter(x='DIST', y='AIR_TIME', ax=ax, alpha=.1, s=1)
)
fig.savefig('c13-scat1.png')


# In[74]:


(flights
    .reset_index(drop=True)
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .pipe(lambda df_:pd.cut(df_.DIST,
          bins=range(0, 2001, 250)))
    .value_counts()
    .sort_index()
)


# In[75]:


zscore = lambda x: (x - x.mean()) / x.std()
short = (flights
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .reset_index(drop=True)    
    .assign(BIN=lambda df_:pd.cut(df_.DIST,
        bins=range(0, 2001, 250)))
)


# In[76]:


scores = (short
    .groupby('BIN')
    ['AIR_TIME']
    .transform(zscore)
)  


# In[77]:


(short.assign(SCORE=scores))


# In[78]:


fig, ax = plt.subplots(figsize=(10,6))    
(short.assign(SCORE=scores)
    .pivot(columns='BIN')
    ['SCORE']
    .plot.box(ax=ax)
)
ax.set_title('Z-Scores for Distance Groups')
fig.savefig('c13-box2.png')


# In[79]:


mask = (short
    .assign(SCORE=scores)
    .pipe(lambda df_:df_.SCORE.abs() >6)
)


# In[80]:


outliers = (flights
    [['DIST', 'AIR_TIME']] 
    .query('DIST <= 2000')
    .dropna()
    .reset_index(drop=True)
    [mask]
    .assign(PLOT_NUM=lambda df_:range(1, len(df_)+1))
)


# In[81]:


outliers


# In[82]:


fig, ax = plt.subplots(figsize=(8,6))
(short
    .assign(SCORE=scores)
    .plot.scatter(x='DIST', y='AIR_TIME',
                  alpha=.1, s=1, ax=ax,
                  table=outliers)
)
outliers.plot.scatter(x='DIST', y='AIR_TIME',
    s=25, ax=ax, grid=True)
outs = outliers[['AIR_TIME', 'DIST', 'PLOT_NUM']]
for t, d, n in outs.itertuples(index=False):
    ax.text(d + 5, t + 5, str(n))
plt.setp(ax.get_xticklabels(), y=.1)
plt.setp(ax.get_xticklines(), visible=False)
ax.set_xlabel('')
ax.set_title('Flight Time vs Distance with Outliers')
fig.savefig('c13-scat3.png', dpi=300, bbox_inches='tight')


# ### How it works...

# ## Stacking area charts to discover emerging trends

# ### How to do it...

# In[83]:


meetup = pd.read_csv('data/meetup_groups.csv',
    parse_dates=['join_date'],
    index_col='join_date')
meetup


# In[84]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
)


# In[85]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
)


# In[86]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
    .cumsum()
)


# In[87]:


(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
    .cumsum()
    .pipe(lambda df_: df_.div(
          df_.sum(axis='columns'), axis='index'))
)


# In[88]:


fig, ax = plt.subplots(figsize=(18,6))    
(meetup
    .groupby([pd.Grouper(freq='W'), 'group']) 
    .size()
    .unstack('group', fill_value=0)
    .cumsum()
    .pipe(lambda df_: df_.div(
          df_.sum(axis='columns'), axis='index'))
    .plot.area(ax=ax,
          cmap='Greys', xlim=('2013-6', None),
          ylim=(0, 1), legend=False)
)
ax.figure.suptitle('Houston Meetup Groups', size=25)
ax.set_xlabel('')
ax.yaxis.tick_right()
kwargs = {'xycoords':'axes fraction', 'size':15}
ax.annotate(xy=(.1, .7), s='R Users',
    color='w', **kwargs)
ax.annotate(xy=(.25, .16), s='Data Visualization',
    color='k', **kwargs)
ax.annotate(xy=(.5, .55), s='Energy Data Science',
    color='k', **kwargs)
ax.annotate(xy=(.83, .07), s='Data Science',
    color='k', **kwargs)
ax.annotate(xy=(.86, .78), s='Machine Learning',
    color='w', **kwargs)
fig.savefig('c13-stacked1.png')


# ### How it works...

# ## Understanding the differences between seaborn and pandas

# ### How to do it...

# In[89]:


employee = pd.read_csv('data/employee.csv',
    parse_dates=['HIRE_DATE', 'JOB_DATE'])
employee


# In[90]:


import seaborn as sns


# In[91]:


fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(y='DEPARTMENT', data=employee, ax=ax)     
fig.savefig('c13-sns1.png', dpi=300, bbox_inches='tight')


# In[92]:


fig, ax = plt.subplots(figsize=(8, 6))
(employee
    ['DEPARTMENT']
    .value_counts()
    .plot.barh(ax=ax)
)
fig.savefig('c13-sns2.png', dpi=300, bbox_inches='tight')


# In[93]:


fig, ax = plt.subplots(figsize=(8, 6))    
sns.barplot(y='RACE', x='BASE_SALARY', data=employee, ax=ax)
fig.savefig('c13-sns3.png', dpi=300, bbox_inches='tight')


# In[94]:


fig, ax = plt.subplots(figsize=(8, 6))    
(employee
    .groupby('RACE', sort=False) 
    ['BASE_SALARY']
    .mean()
    .plot.barh(rot=0, width=.8, ax=ax)
)
ax.set_xlabel('Mean Salary')
fig.savefig('c13-sns4.png', dpi=300, bbox_inches='tight')


# In[95]:


fig, ax = plt.subplots(figsize=(18, 6))        
sns.barplot(x='RACE', y='BASE_SALARY', hue='GENDER',
    ax=ax, data=employee, palette='Greys',
    order=['Hispanic/Latino', 
           'Black or African American',
           'American Indian or Alaskan Native',
           'Asian/Pacific Islander', 'Others',
           'White'])
fig.savefig('c13-sns5.png', dpi=300, bbox_inches='tight')


# In[96]:


fig, ax = plt.subplots(figsize=(18, 6))            
(employee
    .groupby(['RACE', 'GENDER'], sort=False) 
    ['BASE_SALARY']
    .mean()
    .unstack('GENDER')
    .sort_values('Female')
    .plot.bar(rot=0, ax=ax,
        width=.8, cmap='viridis')
)
fig.savefig('c13-sns6.png', dpi=300, bbox_inches='tight')


# In[97]:


fig, ax = plt.subplots(figsize=(8, 6))            
sns.boxplot(x='GENDER', y='BASE_SALARY', data=employee,
            hue='RACE', palette='Greys', ax=ax)
fig.savefig('c13-sns7.png', dpi=300, bbox_inches='tight')


# In[98]:


fig, axs = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
for g, ax in zip(['Female', 'Male'], axs):
    (employee
        .query('GENDER == @g')
        .assign(RACE=lambda df_:df_.RACE.fillna('NA'))
        .pivot(columns='RACE')
        ['BASE_SALARY']
        .plot.box(ax=ax, rot=30)
    )
    ax.set_title(g + ' Salary')
    ax.set_xlabel('')
fig.savefig('c13-sns8.png', bbox_inches='tight')


# ### How it works...

# ## Multivariate analysis with seaborn Grids

# ### How to do it...

# In[99]:


emp = pd.read_csv('data/employee.csv',
    parse_dates=['HIRE_DATE', 'JOB_DATE'])
def yrs_exp(df_):
    days_hired = pd.to_datetime('12-1-2016') - df_.HIRE_DATE
    return days_hired.dt.days / 365.25


# In[100]:


emp = (emp
    .assign(YEARS_EXPERIENCE=yrs_exp)
)


# In[101]:


emp[['HIRE_DATE', 'YEARS_EXPERIENCE']]


# In[102]:


fig, ax = plt.subplots(figsize=(8, 6))        
sns.regplot(x='YEARS_EXPERIENCE', y='BASE_SALARY',
    data=emp, ax=ax)
fig.savefig('c13-scat4.png', dpi=300, bbox_inches='tight')


# In[103]:


grid = sns.lmplot(x='YEARS_EXPERIENCE', y='BASE_SALARY', 
    hue='GENDER', palette='Greys',
    scatter_kws={'s':10}, data=emp)
grid.fig.set_size_inches(8, 6) 
grid.fig.savefig('c13-scat5.png', dpi=300, bbox_inches='tight')


# In[104]:


grid = sns.lmplot(x='YEARS_EXPERIENCE', y='BASE_SALARY',
                  hue='GENDER', col='RACE', col_wrap=3,
                  palette='Greys', sharex=False,
                  line_kws = {'linewidth':5},
                  data=emp)
grid.set(ylim=(20000, 120000))     
grid.fig.savefig('c13-scat6.png', dpi=300, bbox_inches='tight')


# ### How it works...

# ### There's more...

# In[105]:


deps = emp['DEPARTMENT'].value_counts().index[:2]
races = emp['RACE'].value_counts().index[:3]
is_dep = emp['DEPARTMENT'].isin(deps)
is_race = emp['RACE'].isin(races)    
emp2 = (emp
    [is_dep & is_race]
    .assign(DEPARTMENT=lambda df_:
            df_['DEPARTMENT'].str.extract('(HPD|HFD)',
                                    expand=True))
)


# In[106]:


emp2.shape


# In[107]:


emp2['DEPARTMENT'].value_counts()


# In[108]:


emp2['RACE'].value_counts()


# In[109]:


common_depts = (emp
    .groupby('DEPARTMENT') 
    .filter(lambda group: len(group) > 50)
)


# In[110]:


fig, ax = plt.subplots(figsize=(8, 6))   
sns.violinplot(x='YEARS_EXPERIENCE', y='GENDER',
    data=common_depts)
fig.savefig('c13-vio1.png', dpi=300, bbox_inches='tight')


# In[111]:


grid = sns.catplot(x='YEARS_EXPERIENCE', y='GENDER',
                      col='RACE', row='DEPARTMENT',
                      height=3, aspect=2,
                      data=emp2, kind='violin')
grid.fig.savefig('c13-vio2.png', dpi=300, bbox_inches='tight')


# ## Uncovering Simpson's Paradox in the diamonds dataset with seaborn

# ### How to do it...

# In[112]:


dia = pd.read_csv('data/diamonds.csv')
dia


# In[113]:


cut_cats = ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal']
color_cats = ['J', 'I', 'H', 'G', 'F', 'E', 'D']
clarity_cats = ['I1', 'SI2', 'SI1', 'VS2',
                'VS1', 'VVS2', 'VVS1', 'IF']
dia2 = (dia
    .assign(cut=pd.Categorical(dia['cut'], 
                 categories=cut_cats,
                 ordered=True),
            color=pd.Categorical(dia['color'], 
                 categories=color_cats,
                 ordered=True),
            clarity=pd.Categorical(dia['clarity'], 
                 categories=clarity_cats,
                 ordered=True))
)


# In[114]:


dia2


# In[115]:


import seaborn as sns
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4))
sns.barplot(x='color', y='price', data=dia2, ax=ax1)
sns.barplot(x='cut', y='price', data=dia2, ax=ax2)
sns.barplot(x='clarity', y='price', data=dia2, ax=ax3)
fig.suptitle('Price Decreasing with Increasing Quality?')
fig.savefig('c13-bar4.png', dpi=300, bbox_inches='tight')


# In[116]:


grid = sns.catplot(x='color', y='price', col='clarity',
    col_wrap=4, data=dia2, kind='bar')
grid.fig.savefig('c13-bar5.png', dpi=300, bbox_inches='tight')


# In[117]:


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14,4))
sns.barplot(x='color', y='carat', data=dia2, ax=ax1)
sns.barplot(x='cut', y='carat', data=dia2, ax=ax2)
sns.barplot(x='clarity', y='carat', data=dia2, ax=ax3)
fig.suptitle('Diamond size decreases with quality')
fig.savefig('c13-bar6.png', dpi=300, bbox_inches='tight')


# In[118]:


dia2 = (dia2
    .assign(carat_category=pd.qcut(dia2.carat, 5))
)


# In[119]:


from matplotlib.cm import Greys
greys = Greys(np.arange(50,250,40))
grid = sns.catplot(x='clarity', y='price', data=dia2,
   hue='carat_category', col='color',
   col_wrap=4, kind='point', palette=greys)
grid.fig.suptitle('Diamond price by size, color and clarity',
   y=1.02, size=20)
grid.fig.savefig('c13-bar7.png', dpi=300, bbox_inches='tight')


# ### How it works...

# ### There's more...

# In[122]:


g = sns.PairGrid(dia2, height=5,
    x_vars=["color", "cut", "clarity"],
    y_vars=["price"])
g.map(sns.barplot)
g.fig.suptitle('Replication of Step 3 with PairGrid', y=1.02)
g.fig.savefig('c13-bar8.png', dpi=300, bbox_inches='tight')


# In[ ]:




