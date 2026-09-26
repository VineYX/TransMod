"""
Chicago CTA L  (Nov 2018 - Jan 2019, 3 months)
  data/chicago/metro/cta_daily_2018-11.json
  data/chicago/metro/cta_daily_2018-12.json
  data/chicago/metro/cta_daily_2019-01.json
  data/chicago/metro/cta_stations.json
  processed/chicago/metro_hourly_demand.npy  [T=2208, N] 
  processed/chicago/metro_stations.csv
"""
import numpy as np
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "chicago/metro"
OUT  = ROOT / "processed/chicago"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp('2018-11-01')
END   = pd.Timestamp('2019-02-01')   # exclusive

print(" CTA ...")
with open(RAW / "cta_stations.json") as f:
    raw_stations = json.load(f)

station_records = []
for s in raw_stations:
    loc = s.get('location', {})
    try:
        lat = float(loc.get('latitude', 0))
        lon = float(loc.get('longitude', 0))
    except (TypeError, ValueError):
        lat, lon = None, None
    station_records.append({
        'map_id': s.get('map_id', ''),
        'name':   s.get('station_name', ''),
        'lat':    lat,
        'lon':    lon,
    })

station_df = (pd.DataFrame(station_records)
              .dropna(subset=['lat', 'lon'])
              .drop_duplicates('map_id')
              .reset_index(drop=True))
station_df = station_df[(station_df['lat'] != 0) & (station_df['lon'] != 0)].reset_index(drop=True)
print(f"  CTA : {len(station_df)}")

station_to_idx = {str(m): i for i, m in enumerate(station_df['map_id'])}
N = len(station_df)

print(" CTA ...")
all_daily = []
for fname in ['cta_daily_2018-11.json', 'cta_daily_2018-12.json', 'cta_daily_2019-01.json']:
    p = RAW / fname
    if not p.exists():
        print(f"  WARNING: {fname} not found, skipping")
        continue
    with open(p) as f:
        data = json.load(f)
    all_daily.extend(data)
    print(f"  {fname}: {len(data)} records")

daily_df = pd.DataFrame(all_daily)
daily_df['date'] = pd.to_datetime(daily_df['date'], errors='coerce')
daily_df['rides'] = pd.to_numeric(daily_df['rides'], errors='coerce').fillna(0)
daily_df = daily_df[(daily_df['date'] >= START) & (daily_df['date'] < END)]
print(f"  3-month : {len(daily_df)}")

hourly_pattern = np.array([
    0.005, 0.003, 0.002, 0.002, 0.005, 0.020,
    0.045, 0.080, 0.090, 0.065, 0.050, 0.055,
    0.060, 0.055, 0.050, 0.055, 0.065, 0.080,
    0.075, 0.060, 0.045, 0.035, 0.020, 0.013,
], dtype=np.float32)
hourly_pattern /= hourly_pattern.sum()

hours = pd.date_range(START, END, freq='h', inclusive='left')
T = len(hours)
print(f"   T={T}")

demand = np.zeros((T, N), dtype=np.float32)

for _, row in daily_df.iterrows():
    s_idx = station_to_idx.get(str(row['station_id']), None)
    if s_idx is None:
        continue
    day_rides = float(row['rides'])
    day = row['date']
    day_start_idx = int((day - START).total_seconds() // 3600)
    if day_start_idx < 0 or day_start_idx + 24 > T:
        continue
    demand[day_start_idx:day_start_idx+24, s_idx] += day_rides * hourly_pattern

print(f"   shape: {demand.shape}")
print(f"  : {demand.mean():.2f}, : {demand.max():.2f}, : {(demand>0).mean():.3f}")

np.save(OUT / "metro_hourly_demand.npy", demand)
station_df.to_csv(OUT / "metro_stations.csv", index=False)

print(f"\n[OK] Chicago CTA 3-month   T={T}  N={N}")
