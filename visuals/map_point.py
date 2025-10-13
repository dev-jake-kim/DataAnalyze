import pandas as pd
import matplotlib.pyplot as plt

# CSV 파일 경로
csv_path = '../data/ptrs_summary.csv'

def plot_ptrs_summary(csv_path=csv_path):
    # CSV 파일 읽기
    df = pd.read_csv(csv_path)
    # x, y 좌표 추정 (첫 두 컬럼)
    x = df.iloc[:, 0]
    y = df.iloc[:, 1]
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='red', s=10)
    plt.title('ptrs_summary.csv Points')
    plt.xlabel(df.columns[0])
    plt.ylabel(df.columns[1])
    plt.grid(True)
    plt.show()
    plt.savefig('ptrs_summary_plot.png')

if __name__ == '__main__':
    plot_ptrs_summary()

