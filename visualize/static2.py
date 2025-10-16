import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("../data/origin_data.csv", usecols=['call_date', 'xpos', 'ypos'], encoding='cp949')

df['call_date'] = pd.to_datetime(df['call_date'])


df['call_date'] = pd.to_datetime(df['call_date'], format='%Y-%m-%d %H:%M:%S.%f')
df['month'] = df['call_date'].dt.month
df['day'] = df['call_date'].dt.day
df['hour'] = df['call_date'].dt.hour

unique_days_per_month = df.groupby('month')['day'].unique()
all_unique_days = np.concatenate(unique_days_per_month.values)  # 각 월의 유니크 일들을 모두 합침
day_counts_across_months = pd.Series(all_unique_days).value_counts().sort_index()

# 1~31일 전체 인덱스로 재정렬(존재하지 않는 일은 0으로)
day_counts_across_months = day_counts_across_months.reindex(range(1, 32), fill_value=0).astype(int)

call_by_day = df.groupby('day').size().reindex(range(1, 32), fill_value=0).astype(int)

avg_calls_per_day = call_by_day.divide(day_counts_across_months.replace(0, np.nan))
print(avg_calls_per_day)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 32), avg_calls_per_day, marker='o', linestyle='-', color='skyblue')
plt.title('Average Number of Calls per Day of the Month')
plt.xlabel('Day of the Month')
plt.ylabel('Average Number of Calls')
plt.xticks(range(1, 32))
plt.grid(axis='y')
plt.ylim(bottom=0)  # y축이 0부터 시작하도록 설정
plt.tight_layout()
plt.savefig('num_day.png')
plt.show()