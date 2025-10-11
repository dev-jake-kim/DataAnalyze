from pathlib import Path
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import numpy as np

# The user-provided file path is relative to the script's location.
# This assumes the script is run from the 'analizes' directory.
csv_path = Path('../data') / 'changed_time_format.csv'
try:
    df = pd.read_csv(csv_path, encoding='cp949')
except FileNotFoundError:
    # If the script is run from the root directory, this path will be tried.
    csv_path = Path('data') / 'changed_time_format.csv'
    df = pd.read_csv(csv_path, encoding='cp949')

# 각 row에서 (weekday, date, hour) 추출
datas = []
for idx, data in df.iterrows():
    try:
        dt = datetime.datetime.strptime(data['call_date'], '%Y-%m-%d %H')
        datas.append((dt.weekday(), dt.date(), dt.hour))
    except (ValueError, TypeError):
        continue

# 요일별, 날짜별, 시간별 카운트 집계
# {(weekday, date): [24시간 카운트]}
from collections import defaultdict

weekday_date_hour_counts = defaultdict(lambda: np.zeros(24, dtype=int))
for weekday, date, hour in datas:
    weekday_date_hour_counts[(weekday, date)][hour] += 1

# 요일별로 날짜별 시간대 카운트 리스트 만들기
weekday_hour_counts_by_day = [[] for _ in range(7)]  # 7 x [n_days x 24]
for (weekday, date), hour_counts in weekday_date_hour_counts.items():
    weekday_hour_counts_by_day[weekday].append(hour_counts)

# 요일별 시간대별 평균/표준편차 계산
mean_per_hour_by_weekday = []
std_per_hour_by_weekday = []
for day_counts in weekday_hour_counts_by_day:
    if len(day_counts) > 0:
        arr = np.stack(day_counts)  # shape: (n_days, 24)
        mean_per_hour_by_weekday.append(arr.mean(axis=0))
        std_per_hour_by_weekday.append(arr.std(axis=0))
    else:
        mean_per_hour_by_weekday.append(np.zeros(24))
        std_per_hour_by_weekday.append(np.zeros(24))

# Day of week names in English
weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

fig, axes = plt.subplots(7, 1, figsize=(12, 18), sharex=True)
for i in range(7):
    ax = axes[i]
    ax.errorbar(
        range(24),
        mean_per_hour_by_weekday[i],
        yerr=std_per_hour_by_weekday[i],
        fmt='-o',
        capsize=5,
        label=f'{weekday_names[i]}'
    )
    ax.set_title(f'{weekday_names[i]} - Hourly Data Count')
    ax.set_ylabel('Count')
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel('Hour of Day')
plt.tight_layout()
plt.savefig('statics.png')
plt.show()
