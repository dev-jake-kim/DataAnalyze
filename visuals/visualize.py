from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import datetime

from utils.fouriertransform import *


def get_min_max():
    return df['xpos'].min(), df['xpos'].max(), df['ypos'].min(), df['ypos'].max()

def demand_graph(df):
    df['timestamp'] = df['call_date'].apply(lambda x: datetime.datetime.strptime(x, '%Y-%m-%d %H'))
    df = df.sort_values('timestamp')
    start_time = df['timestamp'].min()
    end_time = df['timestamp'].max()
    print(f"Data from {start_time} to {end_time}")
    total_hour = int((end_time - start_time).total_seconds() // 3600)
    demands = [0] * (total_hour+1)
    for time in df['timestamp']:
        hour_diff = (time - start_time).total_seconds() // 3600
        demands[int(hour_diff)] += 1
    plt.figure(figsize=(100, 10))
    plt.plot(range(total_hour+1), demands, 'o-g')
    plt.xlabel('Hours since ' + start_time.strftime('%Y-%m-%d %H'))
    plt.ylabel('Number of Calls')
    plt.title('Demand Over Time')
    xticks = np.arange(0, total_hour + 1, 24)
    plt.xticks(xticks)
    plt.grid()
    plt.savefig('demand_over_time.png')
    plt.show()

    return demands

def make_hit_map(df):
    """
    xpos, ypos를 좌표평면에 점으로 찍어서 보여주는 함수
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(df['xpos'], df['ypos'], s=2, alpha=0.5)
    plt.xlabel('xpos')
    plt.ylabel('ypos')
    plt.title('Hit Map (Scatter Plot of xpos vs ypos)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    csv_path = Path('../data') / 'essential_columns.csv'
    df = pd.read_csv(csv_path, encoding='cp949')
    make_hit_map(df)

