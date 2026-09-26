"""
Graph construction utilities
"""

import torch
import numpy as np


def construct_zone_grid(lat_range, lon_range, zone_size_km=1.0):
    """
    Construct grid-based zones
    Args:
        lat_range: (lat_min, lat_max)
        lon_range: (lon_min, lon_max)
        zone_size_km: zone size in km (default 1km x 1km)
    Returns:
        zone_centers: [N_z, 2] - (lat, lon) for each zone center
        zone_bounds: [N_z, 4] - (lat_min, lat_max, lon_min, lon_max)
        num_zones: int
    """
    # Approximate: 1 degree ≈ 111 km
    zone_size_deg = zone_size_km / 111.0

    # Create grid
    lat_edges = np.arange(lat_range[0], lat_range[1], zone_size_deg)
    lon_edges = np.arange(lon_range[0], lon_range[1], zone_size_deg)

    zone_centers = []
    zone_bounds = []

    for i in range(len(lat_edges) - 1):
        for j in range(len(lon_edges) - 1):
            lat_min, lat_max = lat_edges[i], lat_edges[i+1]
            lon_min, lon_max = lon_edges[j], lon_edges[j+1]

            # Zone center
            center = [(lat_min + lat_max) / 2, (lon_min + lon_max) / 2]
            zone_centers.append(center)

            # Zone bounds
            zone_bounds.append([lat_min, lat_max, lon_min, lon_max])

    zone_centers = np.array(zone_centers)
    zone_bounds = np.array(zone_bounds)

    return zone_centers, zone_bounds, len(zone_centers)


def assign_station_to_zone(station_coords, zone_bounds):
    """
    Assign each station to its zone
    Args:
        station_coords: [N_s, 2] - (lat, lon)
        zone_bounds: [N_z, 4] - (lat_min, lat_max, lon_min, lon_max)
    Returns:
        assignments: [N_s] - zone index for each station
    """
    N_s = len(station_coords)
    N_z = len(zone_bounds)
    assignments = np.zeros(N_s, dtype=int)

    for i, (lat, lon) in enumerate(station_coords):
        # Find which zone contains this station
        for z, (lat_min, lat_max, lon_min, lon_max) in enumerate(zone_bounds):
            if lat_min <= lat < lat_max and lon_min <= lon < lon_max:
                assignments[i] = z
                break

    return assignments


def construct_od_matrix(trips, station_id_map, num_stations):
    """
    Construct OD (Origin-Destination) flow matrix from trip data
    Args:
        trips: list of (start_station_id, end_station_id, count)
        station_id_map: dict mapping station_id -> index
        num_stations: int
    Returns:
        od_matrix: [N, N] - OD flow matrix
    """
    od_matrix = np.zeros((num_stations, num_stations))

    for start_id, end_id, count in trips:
        if start_id in station_id_map and end_id in station_id_map:
            i = station_id_map[start_id]
            j = station_id_map[end_id]
            od_matrix[i, j] += count

    return od_matrix


def normalize_adjacency(adj):
    """
    Normalize adjacency matrix (symmetric normalization)
    A_norm = D^{-1/2} A D^{-1/2}
    Args:
        adj: [N, N] - adjacency matrix
    Returns:
        adj_norm: [N, N]
    """
    # Add self-loops
    adj = adj + np.eye(adj.shape[0])

    # Degree matrix
    degree = np.sum(adj, axis=1)
    d_inv_sqrt = np.power(degree, -0.5)
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0

    # D^{-1/2}
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)

    # Normalize
    adj_norm = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt

    return adj_norm


def edge_index_to_adj(edge_index, edge_weight, num_nodes):
    """
    Convert edge_index format to dense adjacency matrix
    Args:
        edge_index: [2, E]
        edge_weight: [E]
        num_nodes: int
    Returns:
        adj: [N, N]
    """
    adj = torch.zeros(num_nodes, num_nodes, device=edge_index.device)
    adj[edge_index[0], edge_index[1]] = edge_weight
    return adj


def adj_to_edge_index(adj, threshold=0.0):
    """
    Convert dense adjacency matrix to edge_index format
    Args:
        adj: [N, N]
        threshold: minimum edge weight to keep
    Returns:
        edge_index: [2, E]
        edge_weight: [E]
    """
    adj = adj * (adj > threshold)
    edge_index = adj.nonzero(as_tuple=False).t()
    edge_weight = adj[edge_index[0], edge_index[1]]
    return edge_index, edge_weight
