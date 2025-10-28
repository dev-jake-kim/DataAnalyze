import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Sequence
import argparse


def _read_dates(path: str) -> pd.DataFrame:
    """call_date 컬럼을 파싱해 datetime으로 반환. 에러가 있으면 NaT로 처리 후 제거."""
    try:
        df = pd.read_csv(path, usecols=['call_date'], encoding='cp949')
    except Exception:
        # fallback: 시도해보고 못읽으면 예외 재발생
        df = pd.read_csv(path, usecols=['call_date'])

    df['call_date'] = pd.to_datetime(df['call_date'], errors='coerce')
    df = df.dropna(subset=['call_date'])
    return df


def plot_monthly_avg(crop_path: str,
                     non_crop_path: str,
                     title: str = 'Monthly Average Demand',
                     crop_save_path: str = 'num_months_crop.png',
                     non_crop_save_path: str = 'num_months_non_crop.png',
                     compare_save_path: str = 'num_months_compare.png',
                     months: Optional[Sequence[str]] = None,
                     y_min: Optional[float] = None,
                     y_max: Optional[float] = None,
                     line_width: float = 2.5):
    """월별(연-월) 평균 수요를 계산해 crop, non-crop, 비교 그래프를 각각 저장."""
    dc = _read_dates(crop_path)
    dnc = _read_dates(non_crop_path)

    c_counts = dc['call_date'].dt.normalize().groupby(dc['call_date'].dt.normalize()).size()
    n_counts = dnc['call_date'].dt.normalize().groupby(dnc['call_date'].dt.normalize()).size()

    c_month_avg = c_counts.groupby(c_counts.index.to_period('M')).mean()
    n_month_avg = n_counts.groupby(n_counts.index.to_period('M')).mean()

    idx = c_month_avg.index.union(n_month_avg.index).sort_values()
    if months is not None:
        try:
            idx = pd.PeriodIndex([pd.Period(m, freq='M') for m in months], freq='M')
        except Exception:
            pass

    c_vals = c_month_avg.reindex(idx).fillna(0).values
    n_vals = n_month_avg.reindex(idx).fillna(0).values
    labels = [p.to_timestamp().strftime('%y- %m') for p in idx]
    x = range(len(idx))

    # crop only
    plt.figure(figsize=(10, 6))
    plt.plot(x, c_vals, marker='o', color='tab:green', linewidth=line_width)
    plt.title(title + ' (Crop)')
    plt.xlabel('Month')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels, rotation=45)
    plt.grid()
    plt.tight_layout()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.savefig(crop_save_path)
    plt.close()

    # non-crop only
    plt.figure(figsize=(10, 6))
    plt.plot(x, n_vals, marker='o', color='tab:orange', linewidth=line_width)
    plt.title(title + ' (Non-Crop)')
    plt.xlabel('Month')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels, rotation=45)
    plt.grid()
    plt.tight_layout()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.savefig(non_crop_save_path)
    plt.close()

    # compare
    plt.figure(figsize=(10, 6))
    plt.plot(x, c_vals, marker='o', label='crop', color='tab:green', linewidth=line_width)
    plt.plot(x, n_vals, marker='o', label='non-crop', color='tab:orange', linewidth=line_width)
    plt.title(title)
    plt.xlabel('Month')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels, rotation=45)
    plt.legend()
    plt.grid()
    plt.tight_layout()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.savefig(compare_save_path)
    plt.close()


def plot_weekday_avg(crop_path: str,
                     non_crop_path: str,
                     title: str = 'Average Demand by Weekday',
                     crop_save_path: str = 'num_weekday_crop.png',
                     non_crop_save_path: str = 'num_weekday_non_crop.png',
                     compare_save_path: str = 'num_weekday_compare.png',
                     y_min: Optional[float] = None,
                     y_max: Optional[float] = None,
                     line_width: float = 2.5):
    """요일별 평균 수요(월~일)를 crop, non-crop, 비교 그래프로 각각 저장."""
    dc = _read_dates(crop_path)
    dnc = _read_dates(non_crop_path)

    c_counts = dc['call_date'].dt.normalize().groupby(dc['call_date'].dt.normalize()).size()
    n_counts = dnc['call_date'].dt.normalize().groupby(dnc['call_date'].dt.normalize()).size()

    c_wd = c_counts.groupby(c_counts.index.dayofweek).mean().reindex(range(7), fill_value=0)
    n_wd = n_counts.groupby(n_counts.index.dayofweek).mean().reindex(range(7), fill_value=0)

    labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    x = range(7)

    # crop only
    plt.figure(figsize=(10, 6))
    plt.plot(x, c_wd.values, marker='o', color='tab:green', linewidth=line_width)
    plt.title(title + ' (Crop)')
    plt.xlabel('Weekday')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels)
    plt.grid(axis='y')
    plt.tight_layout()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.savefig(crop_save_path)
    plt.close()

    # non-crop only
    plt.figure(figsize=(10, 6))
    plt.plot(x, n_wd.values, marker='o', color='tab:orange', linewidth=line_width)
    plt.title(title + ' (Non-Crop)')
    plt.xlabel('Weekday')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels)
    plt.grid(axis='y')
    plt.tight_layout()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.savefig(non_crop_save_path)
    plt.close()

    # compare
    plt.figure(figsize=(10, 6))
    plt.plot(x, c_wd.values, marker='o', label='crop', color='tab:green', linewidth=line_width)
    plt.plot(x, n_wd.values, marker='o', label='non-crop', color='tab:orange', linewidth=line_width)
    plt.title(title)
    plt.xlabel('Weekday')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels)
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.savefig(compare_save_path)
    plt.close()


def plot_hourly_avg(crop_path: str,
                    non_crop_path: str,
                    title: str = 'Average Demand by Hour',
                    crop_save_path: str = 'num_hourly_crop.png',
                    non_crop_save_path: str = 'num_hourly_non_crop.png',
                    compare_save_path: str = 'num_hourly_compare.png',
                    line_width: float = 2.5):
    """시간대별 평균 수요(0~23시)를 crop, non-crop, 비교 그래프로 각각 저장."""
    dc = _read_dates(crop_path)
    dnc = _read_dates(non_crop_path)

    c_grp = dc.assign(date=dc['call_date'].dt.normalize(), hour=dc['call_date'].dt.hour)
    n_grp = dnc.assign(date=dnc['call_date'].dt.normalize(), hour=dnc['call_date'].dt.hour)

    c_counts = c_grp.groupby(['date', 'hour']).size().reset_index(name='count')
    n_counts = n_grp.groupby(['date', 'hour']).size().reset_index(name='count')

    c_hour_avg = c_counts.groupby('hour')['count'].mean().reindex(range(24), fill_value=0)
    n_hour_avg = n_counts.groupby('hour')['count'].mean().reindex(range(24), fill_value=0)

    x = range(24)
    labels = [f"{h:02d}" for h in x]

    # crop only
    plt.figure(figsize=(12, 6))
    plt.plot(x, c_hour_avg.values, marker='o', color='tab:green', linewidth=line_width)
    plt.title(title + ' (Crop)')
    plt.ylim(bottom=0, top=200)
    plt.xlabel('Hour')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels, rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig(crop_save_path)
    plt.close()

    # non-crop only
    plt.figure(figsize=(12, 6))
    plt.plot(x, n_hour_avg.values, marker='o', color='tab:orange', linewidth=line_width)
    plt.title(title + ' (Non-Crop)')
    plt.xlabel('Hour')
    plt.ylim(bottom=0, top=200)
    plt.ylabel('Average Demand')
    plt.xticks(x, labels, rotation=45)
    plt.grid()
    plt.tight_layout()
    plt.savefig(non_crop_save_path)
    plt.close()

    # compare
    plt.figure(figsize=(12, 6))
    plt.ylim(bottom=0, top=200)
    plt.plot(x, c_hour_avg.values, marker='o', label='crop', color='tab:green', linewidth=line_width)
    plt.plot(x, n_hour_avg.values, marker='o', label='non-crop', color='tab:orange', linewidth=line_width)
    plt.title(title)
    plt.xlabel('Hour')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels, rotation=45)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(compare_save_path)
    plt.close()


def plot_day_of_month_avg(crop_path: str,
                          non_crop_path: str,
                          title: str = 'Average Demand by Day of Month',
                          crop_save_path: str = 'num_day_crop.png',
                          non_crop_save_path: str = 'num_day_non_crop.png',
                          compare_save_path: str = 'num_day_compare.png',
                          y_min: Optional[float] = None,
                          y_max: Optional[float] = None,
                          line_width: float = 2.5):
    """1~31일에 대한 평균 수요를 crop, non-crop, 비교 그래프로 각각 저장."""
    dc = _read_dates(crop_path)
    dnc = _read_dates(non_crop_path)

    dc = dc.assign(month=dc['call_date'].dt.month, day=dc['call_date'].dt.day)
    dnc = dnc.assign(month=dnc['call_date'].dt.month, day=dnc['call_date'].dt.day)

    c_md = dc.groupby(['month', 'day']).size().reset_index(name='count')
    n_md = dnc.groupby(['month', 'day']).size().reset_index(name='count')

    c_day_avg = c_md.groupby('day')['count'].mean().reindex(range(1, 32), fill_value=0)
    n_day_avg = n_md.groupby('day')['count'].mean().reindex(range(1, 32), fill_value=0)

    x = range(1, 32)
    labels = [str(d) for d in x]

    # crop only
    plt.figure(figsize=(12, 6))
    plt.plot(x, c_day_avg.values, marker='o', color='tab:green', linewidth=line_width)
    plt.title(title + ' (Crop)')
    plt.xlabel('Day of Month')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels)
    plt.grid(axis='y')
    plt.tight_layout()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.savefig(crop_save_path)
    plt.close()

    # non-crop only
    plt.figure(figsize=(12, 6))
    plt.plot(x, n_day_avg.values, marker='o', color='tab:orange', linewidth=line_width)
    plt.title(title + ' (Non-Crop)')
    plt.xlabel('Day of Month')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels)
    plt.grid(axis='y')
    plt.tight_layout()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.savefig(non_crop_save_path)
    plt.close()

    # compare
    plt.figure(figsize=(12, 6))
    plt.plot(x, c_day_avg.values, marker='o', label='crop', color='tab:green', linewidth=line_width)
    plt.plot(x, n_day_avg.values, marker='o', label='non-crop', color='tab:orange', linewidth=line_width)
    plt.title(title)
    plt.xlabel('Day of Month')
    plt.ylabel('Average Demand')
    plt.xticks(x, labels)
    plt.grid(axis='y')
    plt.legend()
    plt.tight_layout()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.savefig(compare_save_path)
    plt.close()


def plot_daily_comparison(crop_path: str,
                          non_crop_path: str,
                          title: str = 'Daily Demand Comparison',
                          crop_save_path: str = 'num_daily_crop.png',
                          non_crop_save_path: str = 'num_daily_non_crop.png',
                          compare_save_path: str = 'num_daily_compare.png',
                          start: Optional[str] = '2024-10-01',
                          end: Optional[str] = '2025-03-31',
                          figsize: tuple = (24, 6),
                          y_min: Optional[float] = None,
                          y_max: Optional[float] = None,
                          line_width: float = 2.5):
    """주어진 날짜 범위에 대해 crop, non-crop, 비교 그래프를 각각 저장."""
    dc = _read_dates(crop_path)
    dnc = _read_dates(non_crop_path)

    c_counts = dc['call_date'].dt.normalize().groupby(dc['call_date'].dt.normalize()).size()
    n_counts = dnc['call_date'].dt.normalize().groupby(dnc['call_date'].dt.normalize()).size()

    if start is None:
        start_dt = min(c_counts.index.min(), n_counts.index.min())
    else:
        start_dt = pd.to_datetime(start)
    if end is None:
        end_dt = max(c_counts.index.max(), n_counts.index.max())
    else:
        end_dt = pd.to_datetime(end)

    idx = pd.date_range(start=start_dt.normalize(), end=end_dt.normalize(), freq='D')

    c_vals = c_counts.reindex(idx, fill_value=0).values
    n_vals = n_counts.reindex(idx, fill_value=0).values

    month_change_idx = []
    month_labels = []
    dt_list = list(idx)
    for i, d in enumerate(dt_list):
        if i == 0 or d.month != dt_list[i - 1].month:
            month_change_idx.append(i)
            month_labels.append(d.strftime('%y-%m'))
    x = range(len(idx))

    # crop only
    plt.figure(figsize=figsize)
    plt.plot(x, c_vals, marker='o', markersize=3, color='tab:green', linewidth=line_width)
    plt.title(title + ' (Crop)')
    plt.xlabel('Date')
    plt.ylabel('Average Demand')
    plt.xticks(month_change_idx, month_labels, rotation=45, fontsize=9)
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.tight_layout()
    plt.savefig(crop_save_path)
    plt.close()

    # non-crop only
    plt.figure(figsize=figsize)
    plt.plot(x, n_vals, marker='o', markersize=3, color='tab:orange', linewidth=line_width)
    plt.title(title + ' (Non-Crop)')
    plt.xlabel('Date')
    plt.ylabel('Average Demand')
    plt.xticks(month_change_idx, month_labels, rotation=45, fontsize=9)
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.tight_layout()
    plt.savefig(non_crop_save_path)
    plt.close()

    # compare
    plt.figure(figsize=figsize)
    plt.plot(x, c_vals, marker='o', markersize=3, label='crop', color='tab:green', linewidth=line_width)
    plt.plot(x, n_vals, marker='o', markersize=3, label='non-crop', color='tab:orange', linewidth=line_width)
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Average Demand')
    plt.xticks(month_change_idx, month_labels, rotation=45, fontsize=9)
    plt.legend()
    if y_min is not None or y_max is not None:
        if y_min is not None and y_max is not None:
            plt.ylim(y_min, y_max)
        elif y_min is not None:
            plt.ylim(bottom=y_min)
        else:
            plt.ylim(top=y_max)
    plt.tight_layout()
    plt.savefig(compare_save_path)
    plt.close()


if __name__ == '__main__':
    # 기본 실행: workspace의 data 폴더에 있는 파일을 사용하여 샘플 그래프 생성
    base = '..\\data\\'
    crop = base + 'crop_data.csv'
    non_crop = base + 'non_crop_data.csv'
    y_min = 0
    y_max = 3500

    parser = argparse.ArgumentParser(description='Generate demand plots (crop vs non-crop)')
    parser.add_argument('--line-width', type=float, default=4.0, help='Line width for plots (default: 4.0)')
    args = parser.parse_args()
    lw = args.line_width

    plot_monthly_avg(crop, non_crop, title='Average Demand by Month',
                     crop_save_path='num_months_crop.png',
                     non_crop_save_path='num_months_non_crop.png',
                     compare_save_path='num_months_compare.png',
                     y_max= y_max, y_min=y_min, line_width=lw)
    plot_weekday_avg(crop, non_crop, title='Average Demand by Day of Week',
                     crop_save_path='num_weekday_crop.png',
                     non_crop_save_path='num_weekday_non_crop.png',
                     compare_save_path='num_weekday_compare.png',
                     y_max=y_max, y_min=y_min, line_width=lw)
    plot_hourly_avg(crop, non_crop, title='Average Demand by Hour',
                    crop_save_path='num_hourly_crop.png',
                    non_crop_save_path='num_hourly_non_crop.png',
                    compare_save_path='num_hourly_compare.png', line_width=lw)
    plot_day_of_month_avg(crop, non_crop, title='Daily Demand of Month',
                         crop_save_path='num_day_crop.png',
                         non_crop_save_path='num_day_non_crop.png',
                         compare_save_path='num_day_compare.png',
                         y_max=y_max, y_min=y_min, line_width=lw)
    plot_daily_comparison(crop, non_crop, title='Demand by day',
                         crop_save_path='num_daily_crop.png',
                         non_crop_save_path='num_daily_non_crop.png',
                         compare_save_path='num_daily_compare.png',
                         start='2024-10-01', end='2025-03-31', figsize=(24, 6), y_max=y_max, y_min=y_min, line_width=lw)
