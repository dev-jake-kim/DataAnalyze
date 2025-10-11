from pathlib import Path
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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


datas = []
for idx, data in df.iterrows():
    try:
        dt = datetime.datetime.strptime(data['call_date'], '%Y-%m-%d %H')
        datas.append((data['xpos'], data['ypos'], dt))
    except (ValueError, TypeError):
        continue

datas.sort(key=lambda x: x[2])

timesteps = {}
if datas:
    for x, y, t in datas:
        # Group data into 2-hour intervals
        interval_start_hour = (t.hour // 2) * 2
        timestep_key_dt = t.replace(hour=interval_start_hour, minute=0, second=0, microsecond=0)
        timestep_key = timestep_key_dt.strftime('%Y-%m-%d %H:%M')

        if timestep_key not in timesteps:
            timesteps[timestep_key] = []
        timesteps[timestep_key].append((x, y))

sorted_timesteps = sorted(timesteps.keys())
animation_data = [timesteps[t] for t in sorted_timesteps]

fig, ax = plt.subplots()
all_x = [p[0] for points in animation_data for p in points if points]
all_y = [p[1] for points in animation_data for p in points if points]

if all_x and all_y:
    ax.set_xlim(min(all_x) - 10, max(all_x) + 10)
    ax.set_ylim(min(all_y) - 10, max(all_y) + 10)
else:
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

ax.set_xlabel("xpos")
ax.set_ylabel("ypos")
scat = ax.scatter([], [])
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def update(frame):
    points = animation_data[frame]
    if points:
        scat.set_offsets(points)
    else:
        scat.set_offsets([]) # Clear points for empty intervals

    start_time_str = sorted_timesteps[frame]
    start_time = datetime.datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
    end_time = start_time + datetime.timedelta(hours=2)
    end_time_str = end_time.strftime('%H:%M')
    
    time_text.set_text(f'Time: {start_time_str} - {end_time_str}')
    return scat, time_text

if animation_data:
    ani = FuncAnimation(fig, update, frames=len(animation_data), blit=True, repeat=False)
    
    # Save the animation in the same directory as the script
    output_path = Path(__file__).parent / 'hitmap_animation_2h.gif'
    ani.save(output_path, writer='pillow', fps=20)
    print(f"Animation saved to {output_path}")
else:
    print("No data available to create animation.")

plt.close(fig)
