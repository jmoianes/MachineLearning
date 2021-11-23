# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 23:21:50 2021

@author: JulioMoyanoGarcía

PANDAS - COMPARISION WITH SQL
https://pandas.pydata.org/pandas-docs/stable/getting_started/comparison/comparison_with_sql.html#compare-with-sql

"""



import pandas as pd
import numpy as np

url = ("https://raw.github.com/pandas-dev/pandas/master/pandas/tests/io/data/csv/tips.csv")

dataset = pd.read_csv(url)
tips = dataset

# =============================================================================
# SELECT
# 
# SELECT total_bill, tip, smoker, time
# FROM tips;
#
# =============================================================================

tips[['total_bill', 'tip', 'smoker', 'time']]

# =============================================================================
# SELECT
# 
# SELECT *, tip/total_bill as tip_rate
# FROM tips;
#
# =============================================================================

tips.assign(tip_rate=tips['tip']/tips['total_bill'])

# =============================================================================
# WHERE (filtering in SQL)
# 
# SELECT *
# FROM tips
# WHERE time = 'Dinner';
#
# =============================================================================

tips[tips['time'] == 'Dinner']

is_dinner = tips['time'] == 'Dinner'
is_dinner.value_counts()

tips[is_dinner]

# =============================================================================
# WHERE (filtering in SQL)
# 
# SELECT *
# FROM tips
# WHERE time = 'Dinner' AND tip > 5.00;
#
# =============================================================================

tips[(tips['time'] == 'Dinner') & (tips['tip'] > 5)]

where = (tips['time'] == 'Dinner') & (tips['tip'] > 5)
where.value_counts()

tips[where] # the same as tips[(tips['time'] == 'Dinner') & (tips['tip'] > 5)]

# =============================================================================
# WHERE (filtering in SQL)
# 
# SELECT *
# FROM tips
# WHERE size >= 5 OR total_bill > 45;
#
# =============================================================================

tips[(tips['size'] >= 5) | (tips['total_bill'] > 45)]

where = (tips['size'] >= 5) | (tips['total_bill'] > 45)
where.value_counts()

tips[where] # the same as (tips['size'] >= 5) | (tips['total_bill'] > 45)

# =============================================================================
# WHERE (NULL checking)
# 
# SELECT *
# FROM frame
# WHERE col2 IS NULL;
#
# =============================================================================

frame = pd.DataFrame({"col1":["A", "B", np.nan, "C", "D"], "col2":["F", np.nan, "G", "H", "I"]})

frame

frame[frame['col2'].isna()] # is equivalent to frame[frame['col2'].isnull()]

# =============================================================================
# WHERE (NULL checking)
# 
# SELECT *
# FROM frame
# WHERE col1 IS NOT NULL;
#
# =============================================================================

frame[frame['col1'].notna()] # is equivalent to frame[frame['col2'].notnull()]

# =============================================================================
# GROUP BY
# 
# SELECT sex, count(*)
# FROM tips
# GROUP BY sex;
# /* 
# Female    87
# Male      157
# */
#
# =============================================================================

tips.groupby("sex").count()

tips.groupby("sex")["sex"].count()
tips.groupby("sex")["total_bill"].count()

# =============================================================================
# GROUP BY
# 
# SELECT day, AVG(tip), COUNT(*)
# FROM tips
# GROUP BY day;
#
# =============================================================================

tips.groupby('day').agg({'tip': np.mean, 'day': np.size})

# =============================================================================
# GROUP BY
# 
# SELECT smoker, day, AVG(tip), COUNT(*)
# FROM tips
# GROUP BY smoker, day;
#
# =============================================================================

tips.groupby(['smoker', 'day']).agg({'tip': [np.size, np.mean]})

# =============================================================================
# JOIN
# =============================================================================

df1 = pd.DataFrame({'key':['A', 'B', 'C', 'D'], 'value':np.random.randn(4)})
df2 = pd.DataFrame({'key':['B', 'D', 'D', 'E'], 'value':np.random.randn(4)})

df1

df2

# =============================================================================
# JOIN
#
# SELECT *
# FROM df1
# INNER JOIN df2 ON df1.key = df2.key;
#
# =============================================================================

pd.merge(df1, df2, on='key') # inner join by default
pd.merge(df1, df2, on='key', how='inner')

indexed_df2 = df2.set_index('key')

pd.merge(df1, indexed_df2, left_on='key', right_index=True)

# =============================================================================
# LEFT OUTER JOIN
#
# SELECT *
# FROM df1
# LEFT OUTER JOIN df2 ON df1.key = df2.key;
#
# =============================================================================

pd.merge(df1, df2, on='key', how='left')

# =============================================================================
# RIGHT OUTER JOIN
#
# SELECT *
# FROM df1
# RIGHT OUTER JOIN df2 ON df1.key = df2.key;
#
# =============================================================================

pd.merge(df1, df2, on='key', how='right')

# =============================================================================
# FULL OUTER JOIN
#
# SELECT *
# FROM df1
# FULL OUTER JOIN df2 ON df1.key = df2.key;
#
# =============================================================================

pd.merge(df1, df2, on='key', how='outer')

# =============================================================================
# UNION
#
# SELECT city, rank
# FROM df1
# UNION ALL
# SELECT city, rank
# FROM df2;
#
# =============================================================================

df1 = pd.DataFrame({'city':['Chicago', 'San Francisco', 'New York'], 'rank': range(1, 4)})
df2 = pd.DataFrame({'city':['Chicago', 'Boston', 'Los Angeles'], 'rank': [1, 4, 5]})

pd.concat([df1, df2])

# =============================================================================
# UNION (UNION remove duplicate rows)
#
# SELECT city, rank
# FROM df1
# UNION 
# SELECT city, rank
# FROM df2;
#
# =============================================================================

pd.concat([df1, df2]).drop_duplicates()

# =============================================================================
# LIMIT
#
# SELECT *
# FROM tips
# LIMIT 10;
#
# =============================================================================

tips.head(10)

tips.tail(10)

# Pandas equivalents for some SQL analytic and aggregate functions

# =============================================================================
# TOP N rows with offset
#
# SELECT *
# FROM tips
# ORDER BY tip DESC
# LIMIT 10
# OFFSET 5;
#
# =============================================================================

tips.nlargest(10 + 5, columns='tip').tail(10)

# =============================================================================
# TOP N rows per group
#
# SELECT *
# FROM ( SELECT t.*, ROW_NUMBER() OVER (PARTITION BY day ORDER BY total_bill DESC) as rn from tips t)
# WHERE rn < 3
# ORDER BY day, rn;
#
# =============================================================================

( tips.assign(
    rn = tips.sort_values(['total_bill'], ascending = False)
    .groupby(['day'])
    .cumcount()
    + 1
    )
    .query('rn < 3')
    .sort_values(['day', 'rn'])
)
