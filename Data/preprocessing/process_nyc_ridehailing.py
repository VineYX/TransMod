"""
NYC TLC  (Nov 2018 - Jan 2019, 3 months)
 NYC TLC Taxi Zone  (263 zones)
  data/nyc/ridehailing/yellow_tripdata_2018-11.parquet
  data/nyc/ridehailing/yellow_tripdata_2018-12.parquet
  data/nyc/ridehailing/yellow_tripdata_2019-01.parquet
  processed/nyc/ridehailing_hourly_demand.npy  [T=2208, Z] 
  processed/nyc/ridehailing_zones.csv          zone_id, zone_name, lat, lon
"""
import numpy as np
import pandas as pd
from pathlib import Path
import urllib.request
import os

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "nyc/ridehailing"
OUT  = ROOT / "processed/nyc"
OUT.mkdir(parents=True, exist_ok=True)

zone_csv = RAW / "taxi_zone_lookup.csv"
if not zone_csv.exists():
    print(" TLC Taxi Zone ...")
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
    urllib.request.urlretrieve(url, zone_csv)
    print(f"  : {zone_csv}")

zones = pd.read_csv(zone_csv)
print(f"  Taxi zones: {len(zones)}")
print(f"  : {list(zones.columns)}")

borough_centroids = {
    'Manhattan':     (40.7831, -73.9712),
    'Brooklyn':      (40.6782, -73.9442),
    'Queens':        (40.7282, -73.7949),
    'Bronx':         (40.8448, -73.8648),
    'Staten Island': (40.5795, -74.1502),
    'EWR':           (40.6895, -74.1745),
}
zones['lat'] = zones['Borough'].map(lambda b: borough_centroids.get(b, (40.73, -73.95))[0])
zones['lon'] = zones['Borough'].map(lambda b: borough_centroids.get(b, (40.73, -73.95))[1])

zone_centroid_url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
zone_zip = RAW / "taxi_zones.zip"
if not zone_zip.exists():
    try:
        print(" TLC Taxi Zones shapefile...")
        urllib.request.urlretrieve(zone_centroid_url, zone_zip)
        import zipfile, subprocess
        with zipfile.ZipFile(zone_zip) as z:
            z.extractall(RAW / "taxi_zones")
        try:
            import geopandas as gpd
            gdf = gpd.read_file(RAW / "taxi_zones")
            gdf['centroid'] = gdf.geometry.centroid
            gdf = gdf.to_crs(epsg=4326)
            gdf['lat'] = gdf.geometry.centroid.y
            gdf['lon'] = gdf.geometry.centroid.x
            zone_coords = gdf[['LocationID', 'lat', 'lon']].rename(columns={'LocationID': 'LocationID'})
            zones = zones.merge(zone_coords, on='LocationID', how='left', suffixes=('_approx', ''))
            zones['lat'] = zones['lat'].fillna(zones['lat_approx'])
            zones['lon'] = zones['lon'].fillna(zones['lon_approx'])
            print("   shapefile ")
        except ImportError:
            print("  geopandas ,  borough ")
    except Exception as e:
        print(f"  shapefile  ({e}),  borough ")

print("\n NYC TLC  (parquet, 3 months)...")
parts = []
for fname in ['yellow_tripdata_2018-11.parquet',
              'yellow_tripdata_2018-12.parquet',
              'yellow_tripdata_2019-01.parquet']:
    p = RAW / fname
    if not p.exists():
        print(f"  WARNING: {fname} not found, skipping")
        continue
    try:
        chunk = pd.read_parquet(p, columns=['tpep_pickup_datetime', 'PULocationID'])
    except Exception:
        import pyarrow.parquet as pq
        chunk = pq.read_table(p, columns=['tpep_pickup_datetime', 'PULocationID']).to_pandas()
    parts.append(chunk)
    print(f"  Loaded {fname}: {len(chunk):,} rows")

df = pd.concat(parts, ignore_index=True)
print(f"  : {len(df):,}")

col_map = {'tpep_pickup_datetime': 'pickup_dt', 'PULocationID': 'pu_zone'}
df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

df['pickup_dt'] = pd.to_datetime(df['pickup_dt'], errors='coerce')
df = df.dropna(subset=['pickup_dt', 'pu_zone'])
df = df[(df['pickup_dt'] >= '2018-11-01') & (df['pickup_dt'] < '2019-02-01')]
print(f"  3-month : {len(df):,}")

hours = pd.date_range('2018-11-01', '2019-02-01', freq='h', inclusive='left')
T = len(hours)
hour_to_idx = {h: i for i, h in enumerate(hours)}

valid_zones = zones[zones['LocationID'].between(1, 263)].copy()
zone_to_idx = {int(z): i for i, z in enumerate(valid_zones['LocationID'])}
Z = len(valid_zones)
print(f"  Zone: {Z},  T={T}")

df['hour'] = df['pickup_dt'].dt.floor('h')
df['h_idx'] = df['hour'].map(hour_to_idx)
df['z_idx'] = df['pu_zone'].astype(float).astype('Int64').map(zone_to_idx)
df_valid = df.dropna(subset=['h_idx', 'z_idx']).copy()
df_valid['h_idx'] = df_valid['h_idx'].astype(int)
df_valid['z_idx'] = df_valid['z_idx'].astype(int)

demand = np.zeros((T, Z), dtype=np.float32)
np.add.at(demand, (df_valid['h_idx'].values, df_valid['z_idx'].values), 1)

print(f"   shape: {demand.shape}")
print(f"  : {demand.mean():.2f}, : {demand.max():.0f}, : {(demand>0).mean():.3f}")

np.save(OUT / "ridehailing_hourly_demand.npy", demand)
valid_zones[['LocationID', 'Zone', 'Borough', 'lat', 'lon']].to_csv(
    OUT / "ridehailing_zones.csv", index=False)

print(f"\n[OK] NYC TLC ")
print(f"   demand: {demand.shape} -> {OUT}/ridehailing_hourly_demand.npy")
print(f"   zones:  {Z}            -> {OUT}/ridehailing_zones.csv")
