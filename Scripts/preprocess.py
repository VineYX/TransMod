"""
"""

import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from typing import List, Tuple, Optional, Dict
import json
import pickle
from pathlib import Path


class TemporalGraphPreprocessor:
 """"""

 def __init__(self, normalize_features=True, normalize_edges=False):
 """
 

 Args:
 normalize_features: 
 normalize_edges: 
 """
 self.normalize_features = normalize_features
 self.normalize_edges = normalize_edges
 self.feature_mean = None
 self.feature_std = None

 def load_from_dict(self, data_dict: Dict) -> List[Data]:
 """

 Args:
 data_dict: 'snapshots' 

 Returns:
 
 """
 snapshots = []

 for snapshot in data_dict['snapshots']:
 x = torch.tensor(snapshot['node_features'], dtype=torch.float)
 edge_index = torch.tensor(snapshot['edge_index'], dtype=torch.long)

 edge_attr = None
 if 'edge_attr' in snapshot:
 edge_attr = torch.tensor(snapshot['edge_attr'], dtype=torch.float)

 y = None
 if 'y' in snapshot:
 y = torch.tensor(snapshot['y'], dtype=torch.float)

 data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
 snapshots.append(data)

 return snapshots

 def load_from_json(self, json_path: str) -> List[Data]:
 """
 JSON 

 Args:
 json_path: JSON 

 Returns:
 
 """
 with open(json_path, 'r') as f:
 data_dict = json.load(f)
 return self.load_from_dict(data_dict)

 def load_from_pickle(self, pickle_path: str) -> List[Data]:
 """
 pickle 

 Args:
 pickle_path: pickle 

 Returns:
 
 """
 with open(pickle_path, 'rb') as f:
 data = pickle.load(f)
 return data if isinstance(data, list) else self.load_from_dict(data)

 def normalize_node_features(self, snapshots: List[Data]) -> List[Data]:
 """
 (z-score normalization)

 Args:
 snapshots: 

 Returns:
 
 """
 if not self.normalize_features:
 return snapshots

 all_features = torch.cat([data.x for data in snapshots], dim=0)

 self.feature_mean = all_features.mean(dim=0, keepdim=True)
 self.feature_std = all_features.std(dim=0, keepdim=True)

 self.feature_std[self.feature_std == 0] = 1.0

 normalized_snapshots = []
 for data in snapshots:
 normalized_data = data.clone()
 normalized_data.x = (data.x - self.feature_mean) / self.feature_std
 normalized_snapshots.append(normalized_data)

 return normalized_snapshots

 def create_temporal_sequences(
 self,
 snapshots: List[Data],
 window_size: int,
 stride: int = 1,
 future_steps: int = 1
 ) -> List[Tuple[List[Data], Data]]:
 """

 Args:
 snapshots: 
 window_size: 
 stride: 
 future_steps: 

 Returns:
 (, ) 
 """
 sequences = []

 for i in range(0, len(snapshots) - window_size - future_steps + 1, stride):
 input_seq = snapshots[i:i+window_size]
 target = snapshots[i+window_size+future_steps-1]
 sequences.append((input_seq, target))

 return sequences

 def train_val_test_split(
 self,
 data: List,
 train_ratio: float = 0.7,
 val_ratio: float = 0.15,
 test_ratio: float = 0.15
 ) -> Tuple[List, List, List]:
 """

 Args:
 data: 
 train_ratio: 
 val_ratio: 
 test_ratio: 

 Returns:
 (, , )
 """
 assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

 n = len(data)
 train_end = int(n * train_ratio)
 val_end = int(n * (train_ratio + val_ratio))

 train_data = data[:train_end]
 val_data = data[train_end:val_end]
 test_data = data[val_end:]

 return train_data, val_data, test_data

 def add_self_loops(self, snapshots: List[Data]) -> List[Data]:
 """

 Args:
 snapshots: 

 Returns:
 """
 from torch_geometric.utils import add_self_loops as add_loops

 processed_snapshots = []
 for data in snapshots:
 processed_data = data.clone()
 processed_data.edge_index, _ = add_loops(data.edge_index, num_nodes=data.num_nodes)
 processed_snapshots.append(processed_data)

 return processed_snapshots

 def remove_isolated_nodes(self, snapshots: List[Data]) -> List[Data]:
 """

 Args:
 snapshots: 

 Returns:
 """
 processed_snapshots = []

 for data in snapshots:
 connected_nodes = torch.unique(data.edge_index)

 if len(connected_nodes) == data.num_nodes:
 processed_snapshots.append(data)
 continue

 node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(connected_nodes)}

 new_edge_index = torch.zeros_like(data.edge_index)
 for i in range(data.edge_index.size(1)):
 new_edge_index[0, i] = node_map[data.edge_index[0, i].item()]
 new_edge_index[1, i] = node_map[data.edge_index[1, i].item()]

 new_x = data.x[connected_nodes]

 processed_data = Data(x=new_x, edge_index=new_edge_index)
 if data.edge_attr is not None:
 processed_data.edge_attr = data.edge_attr
 if data.y is not None:
 processed_data.y = data.y

 processed_snapshots.append(processed_data)

 return processed_snapshots

 def save_preprocessed_data(self, data: any, save_path: str):
 """

 Args:
 data: 
 save_path: 
 """
 save_path = Path(save_path)
 save_path.parent.mkdir(parents=True, exist_ok=True)

 with open(save_path, 'wb') as f:
 pickle.dump(data, f)

 print(f": {save_path}")


def generate_sample_data(num_snapshots=20, num_nodes=10, num_features=16):
 """

 Args:
 num_snapshots: 
 num_nodes: 
 num_features: 

 Returns:
 
 """
 snapshots = []

 for t in range(num_snapshots):
 x = torch.randn(num_nodes, num_features) + 0.1 * t

 num_edges = np.random.randint(num_nodes, num_nodes * 3)
 edge_index = torch.randint(0, num_nodes, (2, num_edges))

 y = torch.tensor([np.sin(t / 5.0)], dtype=torch.float)

 data = Data(x=x, edge_index=edge_index, y=y)
 snapshots.append(data)

 return snapshots


if __name__ == "__main__":
 print("=" * 60)
 print("")
 print("=" * 60)

 print("\n1. ...")
 snapshots = generate_sample_data(num_snapshots=100, num_nodes=20, num_features=16)
 print(f" {len(snapshots)} ")
 print(f" : {snapshots[0].num_nodes} , {snapshots[0].num_features} ")

 print("\n2. ...")
 preprocessor = TemporalGraphPreprocessor(normalize_features=True)

 print("\n3. ...")
 snapshots = preprocessor.normalize_node_features(snapshots)
 print(f" : {preprocessor.feature_mean[0, :3].numpy()}")
 print(f" : {preprocessor.feature_std[0, :3].numpy()}")

 print("\n4. ...")
 snapshots = preprocessor.add_self_loops(snapshots)
 print(f" : {snapshots[0].num_edges}")

 print("\n5. ...")
 sequences = preprocessor.create_temporal_sequences(
 snapshots,
 window_size=10,
 stride=5,
 future_steps=1
 )
 print(f" {len(sequences)} ")
 print(f" {len(sequences[0][0])} ")

 print("\n6. //...")
 train_seq, val_seq, test_seq = preprocessor.train_val_test_split(
 sequences,
 train_ratio=0.7,
 val_ratio=0.15,
 test_ratio=0.15
 )
 print(f" : {len(train_seq)} ")
 print(f" : {len(val_seq)} ")
 print(f" : {len(test_seq)} ")

 print("\n7. ...")
 save_dir = Path("/home/josh/PycharmProjects/TransMod/data")
 preprocessor.save_preprocessed_data(train_seq, save_dir / "train_sequences.pkl")
 preprocessor.save_preprocessed_data(val_seq, save_dir / "val_sequences.pkl")
 preprocessor.save_preprocessed_data(test_seq, save_dir / "test_sequences.pkl")

 print("\n" + "=" * 60)
 print("")
 print("=" * 60)
 print("\n:")
 print(" from preprocess import TemporalGraphPreprocessor")
 print(" preprocessor = TemporalGraphPreprocessor()")
 print(" snapshots = preprocessor.load_from_json('your_data.json')")
 print("=" * 60)
