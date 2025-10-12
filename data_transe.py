import pandas as pd
from pathlib import Path
from collections import deque
import datetime
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#이 파일은 버릴 데이터를 고르는 용도입니다.

csv_path = Path('data') / 'changed_time_format.csv'
df = pd.read_csv(csv_path, encoding='cp949')
us_range_x = (128950739, 129500739)
us_range_y = (35756099, 35305729)


def print_dataframe():
    # Load the CSV file with cp949 encoding
    print(df)

def print_head():
    # Load the CSV file with cp949 encoding
    print(list(df.columns))

def proceed_first():
    # xpos, ypos가 각각 us_range_x, us_range_y 범위 내에 있는 행만 필터링
    filtered_df = df[
        (df['xpos'] >= us_range_x[0]) &
        (df['xpos'] <= us_range_x[1]) &
        (df['ypos'] >= us_range_y[1]) &
        (df['ypos'] <= us_range_y[0])
    ]
    # proceed.csv로 저장
    filtered_df.to_csv('data/proceed.csv', index=False, encoding='cp949')
    global csv_path
    csv_path = Path('data') / 'proceed.csv'

def eleminate_duplicates():
    #동일 client_id가 10분 이내에 여러번 등장하는 경우 하나만 남기고 제거
    eleminate_idx = []
    queue = deque([])
    for idx, row in df.iterrows():
        client_id = row['clientid']
        timestamp = datetime.datetime.strptime(row['call_date'], '%Y-%m-%d %H:%M:%S.%f')
        while queue and (timestamp - queue[0][1]).total_seconds() > 600:
            queue.popleft()
        if any(client_id == q[0] for q in queue):
            eleminate_idx.append(idx)
        else:
            queue.append((client_id, timestamp))

    df_dedup = df.drop(index=eleminate_idx)
    df_dedup.to_csv('deduplicated_proceed.csv', index=False, encoding='cp949')
    print(eleminate_idx)
    # print(f"Removed {len(eleminate_idx)} duplicate entries.")
    global csv_path
    csv_path = Path('data') / 'deduplicated_proceed.csv'

def get_min_max():
    print(df['xpos'].min(), df['xpos'].max())
    print(df['ypos'].min(), df['ypos'].max())

def visualize_data(point_x=None, point_y=None, s=3):
    plt.figure(figsize=(30, 30))
    plt.scatter(df['xpos'], df['ypos'], alpha=0.5, s=s)
    if point_x and point_y:
        plt.scatter(point_x, point_y, c='blue', s=100, marker='o', label='Middle Point')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Scatter Plot of X and Y Positions')
    plt.grid(True)
    plt.show()


def k_mean_analize():
    """
    k=3부터 8까지 k-평균 군집화를 수행하고, 결과를 시각화하며,
    각 군집의 데이터 수를 출력합니다.
    """
    # 군집화에 사용할 데이터 (xpos, ypos)
    features = df[['xpos', 'ypos']]

    for k in range(3, 9):
        # KMeans 모델 생성 및 학습
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(features)

        # 각 군집에 속한 데이터 수 출력
        print(f"--- k = {k} 일 때의 군집 별 데이터 수 ---")
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        for cluster_num, count in cluster_counts.items():
            print(f"  군집 {cluster_num}: {count}개")
        print("-" * 35)

        # 군집화 결과 시각화
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(df['xpos'], df['ypos'], c=labels, cmap='viridis', alpha=0.6)

        # 군집의 중심점 표시
        centers = kmeans.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='red', s=100, marker='X', label='Centroids')

        plt.title(f'K-Means Clustering (k={k})')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend(*scatter.legend_elements(), title='Clusters')
        plt.grid(True)
        plt.show()

def middle_point():
    xpos_median = df['xpos'].median()
    ypos_median = df['ypos'].median()
    return xpos_median, ypos_median

def del_outliers(x_mean, y_mean, datas, rate=0.005):
    for data in datas:
        dx, dy = abs(data[0] - x_mean), abs(data[1] - y_mean)
        distance = max(dx, dy)
        data[2]  = distance
    datas.sort(key=lambda x: x[2])
    cut_len = int(len(datas) * rate)
    del datas[-cut_len:]


def refine_middle_point(datas):
    x_sum, y_sum = 0, 0
    for data in datas:
        x_sum += data[0]
        y_sum += data[1]
    return x_sum // len(datas), y_sum // len(datas)

def crop(threshold_rate = 0.85):
    datas = []
    for idx, row in df.iterrows():
        datas.append([row['xpos'], row['ypos'], 0, idx])
    cnt = 0
    threshold = int(len(datas) * threshold_rate)
    print(f"Initial Data Count: {len(datas)}, Threshold: {threshold}")
    while len(datas) > threshold:
        cnt += 1
        x_mid, y_mid = refine_middle_point(datas)
        print(f"{cnt}: Current Middle Point: {x_mid}, {y_mid}, Data Count: {len(datas)}")
        del_outliers(x_mid, y_mid, datas, rate=0.001)

    #남아있는 data만 저장
    remaining_indices = [data[3] for data in datas]
    filtered_df = df.loc[remaining_indices]
    filtered_df.to_csv('data/cropped_proceed.csv', index=False, encoding='cp949')
    print(f"Final Data Count: {len(datas)}")
    return refine_middle_point(datas)

def change_time_format():
    #'%Y-%m-%d %H:%M:%S.%f' -> '%Y-%m-%d %H
    df['call_date'] = pd.to_datetime(df['call_date'], format='%Y-%m-%d %H:%M:%S.%f')
    df['call_date'] = df['call_date'].dt.strftime('%Y-%m-%d %H')
    df.to_csv('data/changed_time_format.csv', index=False, encoding='cp949')

def drop_nonessential_columns():
    essential_columns = ['clientid', 'call_date', 'xpos', 'ypos']
    df_filtered = df[essential_columns]
    minX= df['xpos'].min()
    minY= df['ypos'].min()
    df_filtered['xpos'] = df_filtered['xpos'].apply(lambda x: x - minX)
    df_filtered['ypos'] = df_filtered['ypos'].apply(lambda y: y - minY)

    df_filtered.to_csv('data/essential_columns.csv', index=False, encoding='cp949')


if __name__ == "__main__":
    drop_nonessential_columns()
