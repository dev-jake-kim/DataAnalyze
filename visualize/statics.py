import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. 데이터 불러오기
df = pd.read_csv("../data/essential_columns.csv", usecols=['call_date', 'xpos', 'ypos'], encoding='cp949')

df['call_date'] = pd.to_datetime(df['call_date'])


df['call_date'] = pd.to_datetime(df['call_date'], format='%Y-%m-%d %H')
df['call_date'] = df['call_date'].dt.normalize()

english_weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
df['weekday'] = df['call_date'].dt.dayofweek.map(lambda x: english_weekdays[int(x)] if pd.notna(x) else None)
# python
# 정렬 및 결측 제거 후 groupby로 일별 집계 (IndexError 방지)
df.sort_values('call_date', inplace=True)
df = df.dropna(subset=['call_date'])  # NaT가 있으면 제거

counts_per_day = df.groupby('call_date').size().sort_index()
total_days = counts_per_day.size
print(f"Total unique days: {total_days}")
print(df['call_date'].min(), df['call_date'].max())

day_arr = counts_per_day.values

plt.figure(figsize=(10, 6))
plt.plot(range(total_days), day_arr, marker='o')
plt.title('Number of Calls per Day')
plt.xlabel('Days')
plt.ylabel('Number of Calls')
plt.grid()
plt.savefig('num_days.png')
plt.show()


#요일별
counts_per_weekday = df.groupby('weekday').size().reindex(english_weekdays).fillna(0).astype(int)
unique_days_per_weekday = df.groupby('weekday')['call_date'].nunique().reindex(english_weekdays).fillna(0).astype(int)
avg_calls_per_day = counts_per_weekday.divide(unique_days_per_weekday.replace(0, np.nan))
print(avg_calls_per_day)


plt.figure(figsize=(10, 6))
plt.ylim(bottom=1000, top=2500)
plt.bar(english_weekdays, avg_calls_per_day, color='skyblue')
plt.title('Average Number of Calls per Day of the Week')
plt.xlabel('Day of the Week')
plt.ylabel('Average Number of Calls')
plt.grid(axis='y')
plt.savefig('num_weekday.png')
plt.show()



