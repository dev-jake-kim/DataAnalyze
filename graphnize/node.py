from typing import Self
import pandas as pd
from pathlib import Path
import datetime
from tqdm import tqdm

class Data:
    def __init__(self, row:pd.Series):
        self.call_date = datetime.datetime.strptime(row['call_date'], '%Y-%m-%d %H')
        self.client_id = row['clientid']

class Node:
    distance_func = lambda a, b: ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
    def __init__(self, _data:Data, position:tuple[int, int]):
        self._position = position
        self._data = _data
        self.edges = []

    def get_position(self):
        return self._position
    def _add_edge(self, node: Self):
        self.edges.append(node)

    def get_data(self):
        return self._data
    def connect(self, node: Self):
        self._add_edge(node)
        node._add_edge(self)
    def get_edges(self):
        return self.edges

    def distance(self, other: Self):
        return Node.distance_func(self._position, other._position)


if __name__ == "__main__":
    csv_path = Path('../data') / 'changed_time_format.csv'
    df = pd.read_csv(csv_path, encoding='cp949')
    nodes = []
    minX, maxX = df['xpos'].min(), df['xpos'].max()
    minY, maxY = df['ypos'].min(), df['ypos'].max()
    print(f"X range: {minX} to {maxX}, Y range: {minY} to {maxY}")
    for idx, row in df.iterrows():
        data = Data(row)
        node = Node(data, (row['xpos'], row['ypos']))
        nodes.append(node)

    print(f"Created {len(nodes)} nodes.")
