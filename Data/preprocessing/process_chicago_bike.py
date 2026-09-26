"""
Chicago Divvy  (Nov 2018 - Jan 2019, 3 months)
  data/chicago/bike/Divvy_Trips_2018_Q4.csv  (Oct-Dec 2018 -> filter Nov-Dec)
  data/chicago/bike/Divvy_Trips_2019_Q1.csv  (Jan-Mar 2019 -> filter Jan)
  processed/chicago/bike_hourly_demand.npy  [T=2208, N]
  processed/chicago/bike_stations.csv
  processed/chicago/bike_od_hourly.npy      [T=2208, N, N]
"""
import numpy as np
import pandas as pd
import json, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
BIKE = ROOT / "chicago/bike"
OUT  = ROOT / "processed/chicago"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp('2018-11-01')
END   = pd.Timestamp('2019-02-01')

parts = []
for fname in ['Divvy_Trips_2018_Q4.csv', 'Divvy_Trips_2019_Q1.csv']:
    p = BIKE / fname
    if not p.exists():
        print(f"  WARNING: {fname} not found, skipping")
        continue
    chunk = pd.read_csv(p, low_memory=False)
    print(f"  Loaded {fname}: {len(chunk):,} rows")
    parts.append(chunk)

df = pd.concat(parts, ignore_index=True)
print(f"  : {len(df):,}")

df['starttime'] = pd.to_datetime(df['start_time'], errors='coerce')
df['stoptime']  = pd.to_datetime(df['end_time'],   errors='coerce')
df = df.dropna(subset=['starttime'])
df = df[(df['starttime'] >= START) & (df['starttime'] < END)]
print(f"  3-month : {len(df):,}")

start_id   = 'from_station_id'
start_name = 'from_station_name'
end_id     = 'to_station_id'
end_name   = 'to_station_name'

full_station_file = BIKE / "divvy_stations_full.json"
if not full_station_file.exists():
    try:
        print(" Divvy ...")
        urllib.request.urlretrieve(
            "https://data.cityofchicago.org/resource/bbyy-e7gq.json?$limit=2000",
            full_station_file)
    except Exception as e:
        print(f"  : {e}")

name_to_coord = {}
if full_station_file.exists():
    try:
        with open(full_station_file) as f:
            od_stations = json.load(f)
        for s in od_stations:
            if 'latitude' in s and 'longitude' in s:
                name_to_coord[s['station_name'].strip()] = (
                    float(s['latitude']), float(s['longitude']))
        print(f"  : {len(name_to_coord)} ")
    except Exception as e:
        print(f"  : {e}")

all_ids = pd.concat([
    df[[start_id, start_name]].rename(columns={start_id: 'id', start_name: 'name'}),
    df[[end_id,   end_name  ]].rename(columns={end_id: 'id',   end_name: 'name'}),
], ignore_index=True).dropna()
all_ids['id'] = all_ids['id'].astype(str).str.strip()

station_df = (all_ids.groupby('id')
              .agg({'name': 'first'})
              .reset_index()
              .sort_values('id')
              .reset_index(drop=True))
station_df['lat'] = station_df['name'].map(lambda n: name_to_coord.get(n, (None,None))[0])
station_df['lon'] = station_df['name'].map(lambda n: name_to_coord.get(n, (None,None))[1])
station_df['lat'] = station_df['lat'].fillna(41.8781)
station_df['lon'] = station_df['lon'].fillna(-87.6298)

station_to_idx = {sid: i for i, sid in enumerate(station_df['id'])}
N = len(station_df)
matched = station_df['lat'].ne(41.8781).sum()
print(f"  : {N}, : {matched} ({100*matched/N:.0f}%)")

hours = pd.date_range(START, END, freq='h', inclusive='left')
T = len(hours)
hour_to_idx = {h: i for i, h in enumerate(hours)}
print(f"   T={T}")

df['start_id_str'] = df[start_id].astype(str).str.strip()
df['end_id_str']   = df[end_id].astype(str).str.strip()
df['h_idx']  = df['starttime'].dt.floor('h').map(hour_to_idx)
df['sh_idx'] = df['stoptime'].dt.floor('h').map(hour_to_idx)
df['s_idx']  = df['start_id_str'].map(station_to_idx)
df['e_idx']  = df['end_id_str'].map(station_to_idx)

pickup  = np.zeros((T, N), dtype=np.float32)
dropoff = np.zeros((T, N), dtype=np.float32)

pu_valid = df.dropna(subset=['h_idx', 's_idx']).copy()
pu_valid['h_idx'] = pu_valid['h_idx'].astype(int)
pu_valid['s_idx'] = pu_valid['s_idx'].astype(int)
np.add.at(pickup, (pu_valid['h_idx'].values, pu_valid['s_idx'].values), 1)

do_valid = df.dropna(subset=['sh_idx', 'e_idx']).copy()
do_valid['sh_idx'] = do_valid['sh_idx'].astype(int)
do_valid['e_idx']  = do_valid['e_idx'].astype(int)
np.add.at(dropoff, (do_valid['sh_idx'].values, do_valid['e_idx'].values), 1)

demand = pickup + dropoff
print(f"   shape: {demand.shape}, : {demand.mean():.2f}")

print(" OD ...")
od_valid = df.dropna(subset=['h_idx', 's_idx', 'e_idx']).copy()
od_valid['h_idx'] = od_valid['h_idx'].astype(int)
od_valid['s_idx'] = od_valid['s_idx'].astype(int)
od_valid['e_idx'] = od_valid['e_idx'].astype(int)
od = np.zeros((T, N, N), dtype=np.float32)
np.add.at(od, (od_valid['h_idx'].values, od_valid['s_idx'].values, od_valid['e_idx'].values), 1)
print(f"  OD shape: {od.shape}")

np.save(OUT / "bike_hourly_demand.npy", demand)
np.save(OUT / "bike_od_hourly.npy", od)
station_df.to_csv(OUT / "bike_stations.csv", index=False)
pd.DataFrame({'hour': [str(h) for h in hours]}).to_csv(OUT / "bike_hours.csv", index=False)

print(f"\n[OK] Chicago Divvy 3-month   T={T}  N={N}")
