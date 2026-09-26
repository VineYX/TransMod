"""
Chicago TNP  (Nov 2018 - Jan 2019, 3 months)
 (per community area per hour)
  data/chicago/ridehailing/tnp_hourly_2018-11.csv
  data/chicago/ridehailing/tnp_hourly_2018-12.csv
  data/chicago/ridehailing/tnp_hourly_2019-01.csv
  processed/chicago/ridehailing_hourly_demand.npy  [T=2208, Z]
  processed/chicago/ridehailing_zones.csv
"""
import numpy as np
import pandas as pd
import json, urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "chicago/ridehailing"
OUT  = ROOT / "processed/chicago"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp('2018-11-01')
END   = pd.Timestamp('2019-02-01')

print(" Chicago TNP  (3 months)...")
parts = []
for fname in ['tnp_hourly_2018-11.csv', 'tnp_hourly_2018-12.csv', 'tnp_hourly_2019-01.csv']:
    p = RAW / fname
    if not p.exists():
        print(f"  WARNING: {fname} not found, skipping")
        continue
    chunk = pd.read_csv(p)
    print(f"  {fname}: {len(chunk):,} rows")
    parts.append(chunk)

df = pd.concat(parts, ignore_index=True)
print(f"  : {len(df):,}")

df['trip_date'] = pd.to_datetime(df['trip_date'], errors='coerce')
df['pickup_dt'] = df['trip_date'] + pd.to_timedelta(df['trip_hour'].astype(int), unit='h')
df = df.dropna(subset=['pickup_dt'])
df = df[(df['pickup_dt'] >= START) & (df['pickup_dt'] < END)]
print(f"  3-month : {len(df):,}")

df['trips'] = pd.to_numeric(df['trips'], errors='coerce').fillna(0).astype(int)

ca_file = RAW / "chicago_community_areas.json"
if not ca_file.exists():
    try:
        print(" Chicago Community Area ...")
        url = "https://data.cityofchicago.org/resource/igwz-8jzy.json?$limit=100"
        urllib.request.urlretrieve(url, ca_file)
    except Exception as e:
        print(f"  : {e}")

ca_coords = {}
if ca_file.exists():
    try:
        with open(ca_file) as f:
            cas = json.load(f)
        for ca in cas:
            ca_num = ca.get('area_num_1', ca.get('area_numbe', ''))
            if 'centroid' in ca:
                c = ca['centroid']
                ca_coords[str(ca_num)] = (float(c.get('latitude', 0)),
                                          float(c.get('longitude', 0)))
            elif 'the_geom' in ca:
                # compute centroid from polygon coordinates [lon, lat]
                geom = ca['the_geom']
                coords_flat = []
                def _flatten(obj):
                    if isinstance(obj, list):
                        if obj and isinstance(obj[0], (int, float)):
                            coords_flat.append(obj)
                        else:
                            for item in obj:
                                _flatten(item)
                _flatten(geom.get('coordinates', []))
                if coords_flat:
                    lons = [c[0] for c in coords_flat]
                    lats = [c[1] for c in coords_flat]
                    ca_coords[str(ca_num)] = (sum(lats)/len(lats), sum(lons)/len(lons))
        print(f"  Community Area : {len(ca_coords)} ")
    except Exception as e:
        print(f"  : {e}")

if len(ca_coords) < 10:
    ca_coords = {str(i): (41.8781, -87.6298) for i in range(1, 78)}

hours = pd.date_range(START, END, freq='h', inclusive='left')
T = len(hours)
hour_to_idx = {h: i for i, h in enumerate(hours)}
print(f"   T={T}")

df['pickup_ca'] = pd.to_numeric(df['pickup_community_area'], errors='coerce')
df = df.dropna(subset=['pickup_ca'])
df['pickup_ca'] = df['pickup_ca'].astype(int)

valid_cas = sorted([ca for ca in df['pickup_ca'].unique() if 1 <= ca <= 77])
ca_to_idx = {ca: i for i, ca in enumerate(valid_cas)}
Z = len(valid_cas)
print(f"  Community Areas: {Z}")

df['h_idx'] = df['pickup_dt'].map(hour_to_idx)
df['z_idx'] = df['pickup_ca'].map(ca_to_idx)
df_valid = df.dropna(subset=['h_idx', 'z_idx']).copy()
df_valid['h_idx'] = df_valid['h_idx'].astype(int)
df_valid['z_idx'] = df_valid['z_idx'].astype(int)

demand = np.zeros((T, Z), dtype=np.float32)
np.add.at(demand, (df_valid['h_idx'].values, df_valid['z_idx'].values),
          df_valid['trips'].values.astype(np.float32))

print(f"   shape: {demand.shape}")
print(f"  : {demand.mean():.2f}, : {demand.max():.0f}, : {(demand>0).mean():.3f}")

zone_df = pd.DataFrame([
    {'zone_id': ca, 'lat': ca_coords.get(str(ca), (41.8781, -87.6298))[0],
                    'lon': ca_coords.get(str(ca), (41.8781, -87.6298))[1]}
    for ca in valid_cas
])

np.save(OUT / "ridehailing_hourly_demand.npy", demand)
zone_df.to_csv(OUT / "ridehailing_zones.csv", index=False)

print(f"\n[OK] Chicago TNP 3-month   T={T}  Z={Z}")
