from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np


def save_graph_json(
    output_path: Path,
    nodes: list[dict],
    demands: np.ndarray,
    near_demands: np.ndarray,
    temporal_features: list[dict],
    od_flows: list[list[dict]],
) -> None:
    print("\n" + "=" * 60)
    print("Saving graph to JSON")
    print("=" * 60)

    x = []
    for time_idx in range(len(demands)):
        x_t = {
            'demand': demands[time_idx].tolist(),
            'near_demands': near_demands[time_idx].tolist(),
            'day': temporal_features[time_idx]['day'],
            'time': temporal_features[time_idx]['time'],
            'holiday': temporal_features[time_idx]['holiday'],
            'OD': od_flows[time_idx],
        }
        x.append(x_t)

    data = {
        'nodes': nodes,
        'x': x,
    }

    json_str = json.dumps(data, indent=2, ensure_ascii=False, separators=(',', ': '))

    def compact_list(match: re.Match[str]) -> str:
        return match.group(0).replace('\n', '').replace(' ', '')

    compact_json = re.sub(r'\[[\s\d,]+\]', compact_list, json_str)

    with open(output_path, 'w', encoding='utf-8') as file:
        file.write(compact_json)

    print(f"\nJSON saved to: {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1024:.2f} KB")
    print(f"\nData structure:")
    print(f"  nodes: {len(nodes)} nodes")
    print(f"  x: {len(x)} timesteps")
    print(f"    - Each timestep has: demand (list of {len(demands[0])}), near_demands, day, time, holiday, OD (list)")
