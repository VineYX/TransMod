"""
Data loader utilities for TransMod
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path


class MobilityDataset(Dataset):
    """
    Dataset for mobility demand forecasting
    """

    def __init__(self, data_path, mode='bike', window_size=12, horizon=1):
        """
        Args:
            data_path: path to data file
            mode: 'bike', 'metro', or 'rh'
            window_size: temporal window size
            horizon: prediction horizon
        """
        self.data_path = data_path
        self.mode = mode
        self.window_size = window_size
        self.horizon = horizon

        # Load data (to be implemented based on actual data format)
        self.load_data()

    def load_data(self):
        """
        Load data from file
        User should implement this based on their data format
        """
        # Placeholder - user to implement
        # Expected format:
        # - demand: [T, N] - demand time series
        # - features: [T, N, F] - station/zone features
        # - coords: [N, 2] - coordinates
        # - od_flows: [T, N, N] - OD flows (optional)

        print(f"Loading {self.mode} data from {self.data_path}...")
        print("NOTE: User needs to implement data loading based on their format")

        # Dummy data for testing
        self.demand = torch.randn(1000, 100)  # [T, N]
        self.features = torch.randn(1000, 100, 6)  # [T, N, F]
        self.coords = torch.randn(100, 2)  # [N, 2]
        self.time_features = torch.randn(1000, 16)  # [T, 16]

        self.num_timesteps = self.demand.shape[0]
        self.num_nodes = self.demand.shape[1]

    def __len__(self):
        """Number of samples"""
        return self.num_timesteps - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        """
        Get one sample
        Returns:
            sample: dict with keys:
                - features: [window_size, N, F]
                - demand_history: [window_size, N]
                - demand_target: [horizon, N]
                - time_features: [window_size, 16]
                - coords: [N, 2]
        """
        # Historical window
        features = self.features[idx:idx+self.window_size]
        demand_history = self.demand[idx:idx+self.window_size]
        time_features = self.time_features[idx:idx+self.window_size]

        # Target (next horizon steps)
        demand_target = self.demand[idx+self.window_size:idx+self.window_size+self.horizon]

        sample = {
            'features': features,
            'demand_history': demand_history,
            'demand_target': demand_target,
            'time_features': time_features,
            'coords': self.coords,
            'idx': idx
        }

        return sample


class ZoneDataset(Dataset):
    """
    Dataset for zone-level demand (after aggregation)
    """

    def __init__(self, zone_data_path, window_size=12, horizon=1):
        """
        Args:
            zone_data_path: path to preprocessed zone data
            window_size: temporal window
            horizon: prediction horizon
        """
        self.zone_data_path = zone_data_path
        self.window_size = window_size
        self.horizon = horizon

        self.load_data()

    def load_data(self):
        """Load zone-level data"""
        print(f"Loading zone data from {self.zone_data_path}...")

        # Placeholder
        self.demand = torch.randn(1000, 50)  # [T, N_z]
        self.spatial_features = torch.randn(50, 128)  # [N_z, 128] - POI+Road+Graph
        self.zone_centers = torch.randn(50, 2)  # [N_z, 2]
        self.time_features = torch.randn(1000, 16)  # [T, 16]

        self.num_timesteps = self.demand.shape[0]
        self.num_zones = self.demand.shape[1]

    def __len__(self):
        return self.num_timesteps - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        demand_history = self.demand[idx:idx+self.window_size]
        demand_target = self.demand[idx+self.window_size:idx+self.window_size+self.horizon]
        time_features = self.time_features[idx:idx+self.window_size]

        return {
            'demand_history': demand_history,
            'demand_target': demand_target,
            'time_features': time_features,
            'spatial_features': self.spatial_features,
            'zone_centers': self.zone_centers,
            'idx': idx
        }


def get_dataloaders(config, mode='pretrain'):
    """
    Create data loaders for training
    Args:
        config: Config object
        mode: 'pretrain' or 'transfer'
    Returns:
        train_loader, val_loader, test_loader
    """
    if mode == 'pretrain':
        # Load bike and metro data
        bike_dataset = MobilityDataset(
            data_path=config.bike_data_path,
            mode='bike',
            window_size=config.time_window
        )
        metro_dataset = MobilityDataset(
            data_path=config.metro_data_path,
            mode='metro',
            window_size=config.time_window
        )

        # For simplicity, assume they have same length
        # In practice, you might need to handle different lengths

        # Create data loaders
        train_loader = DataLoader(
            bike_dataset,  # Combined dataset needed
            batch_size=config.pretrain_batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )

        val_loader = DataLoader(
            bike_dataset,
            batch_size=config.pretrain_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        test_loader = DataLoader(
            bike_dataset,
            batch_size=config.pretrain_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

    else:  # transfer
        rh_dataset = ZoneDataset(
            zone_data_path=config.rh_data_path,
            window_size=config.time_window
        )

        # Split dataset
        train_size = int(config.train_ratio * len(rh_dataset))
        val_size = int(config.val_ratio * len(rh_dataset))
        test_size = len(rh_dataset) - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
            rh_dataset, [train_size, val_size, test_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.transfer_batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.transfer_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.transfer_batch_size,
            shuffle=False,
            num_workers=config.num_workers
        )

    return train_loader, val_loader, test_loader


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save model checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path}")
    return checkpoint['epoch'], checkpoint['metrics']
