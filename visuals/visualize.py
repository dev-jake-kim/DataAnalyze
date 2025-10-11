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




if __name__ == "__main__":
    csv_path = Path('../data') / 'changed_time_format.csv'
    df = pd.read_csv(csv_path, encoding='cp949')
    demands = demand_graph(df)
    t_space = np.arange(len(demands))
    y_space = np.array(demands)

    # 주기(시간 단위): 1~672 (1시간~4주)
    periods = np.arange(1, 24 * 28 + 1)
    frequencies = 1 / periods

    mid_real = []
    mid_imag = []

    for freq in frequencies:
        real, imag = map_to_complex_plane(t_space, y_space, freq)
        mid_real.append(middle_point(real))
        mid_imag.append(middle_point(imag))

    # 결과 시각화
    plt.figure(figsize=(16, 6))
    plt.plot(periods, np.abs(mid_real), label='Real (abs)')
    plt.plot(periods, np.abs(mid_imag), label='Imag (abs)')
    plt.xlabel('Period (hours)')
    plt.ylabel('Mean Value')
    plt.title('Fourier Transform of Demand (1h ~ 4weeks)')
    plt.legend()
    plt.grid()
    plt.savefig('fourier_transform_demand.png')
    plt.show()
