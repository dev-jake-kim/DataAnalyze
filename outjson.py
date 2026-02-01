import pandas as pd
from pathlib import Path
from typing import Dict, Any

class OutJson:
    def __init__(self, minHour, maxHour, self_loop_w=1.0, total_nodes=0, edges=None, demands=None, assignment_matrix=None, clusters_demand=None, clusters_edge=None, dropped_demand=0):
        self.minHour = minHour
        self.maxHour = maxHour
        self.self_loop_w = self_loop_w
        self.total_nodes = total_nodes
        self.edges = edges if edges is not None else []
        self.demands = demands if demands is not None else []
        self.assignment_matrix = assignment_matrix
        self.clusters_demand = clusters_demand
        self.clusters_edge = clusters_edge
        self.dropped_points = dropped_demand


    def to_dict(self) -> Dict[str, Any]:
        # hours 리스트 생성
        total_hours = int((self.maxHour - self.minHour).total_seconds() // 3600) + 1
        hours = []
        for h in range(total_hours):
            hour = self.minHour + pd.Timedelta(hours=h)
            hours.append(hour.strftime("%Y-%m-%dT%H:%M:%S"))
        return {
            "meta": {
                "hours": hours,
                "self_loop_w": self.self_loop_w,
                "edge_weight": "geom_mean",
                "dropped_points": self.dropped_points
            },
            "nodes": [{"id": i} for i in range(self.total_nodes)],
            "edges": [
                {"u": int(e[0]), "v": int(e[1]), "w_dist": float(e[2]), "w_dtw": float(e[3])}
                for e in self.edges
            ],
            "x": self.demands,
            "cluster_assignment": self.assignment_matrix,
            "clusters_demands": self.clusters_demand,
            "clusters_edges": [
                {"u": int(e[0]), "v": int(e[1]), "w_dist": float(e[2]), "w_dtw": float(e[3])}
                for e in self.clusters_edge
            ]
        }

    def save_json(self, path: Path):
        import json
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(
                self.to_dict(),
                f,
                ensure_ascii=False,
                separators=(',', ':')  # 공백 제거(콤마/콜론 뒤 공백 없음)
            )