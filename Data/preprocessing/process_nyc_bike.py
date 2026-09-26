"""
NYC Citi Bike  (Nov 2018 - Jan 2019, 3 months)
: data/nyc/bike/201811/12/201901-citibike-tripdata.csv
  processed/nyc/bike_hourly_demand.npy  [T, N]
  processed/nyc/bike_stations.csv
  processed/nyc/bike_od_hourly.npy      [T, N, N]
  processed/nyc/bike_hours.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT  = Path(__file__).parent.parent
RAW   = ROOT / "nyc/bike"
OUT   = ROOT / "processed/nyc"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp('2018-11-01')
END   = pd.Timestamp('2019-02-01')   # exclusive

import zipfile as _zf

def _load_month(fname):
    """Load a monthly CSV, searching individual file first then annual zips."""
    p = RAW / fname
    if p.exists():
        df = pd.read_csv(p, low_memory=False)
        print(f"  Loaded {fname}: {len(df):,} rows")
        return df
    # search annual zips; files may be split as _1.csv, _2.csv, etc.
    year = fname[:4]
    stem = fname[:-4]  # e.g. '201901-citibike-tripdata'
    for zname in [f'{year}-citibike-tripdata.zip', f'{fname[:-4]}.csv.zip']:
        zp = RAW / zname
        if not zp.exists():
            continue
        with _zf.ZipFile(zp) as z:
            # Match exact name or split parts (_1.csv, _2.csv, ...)
            candidates = sorted([n for n in z.namelist()
                                  if stem in n and n.endswith('.csv')
                                  and not n.startswith('__MACOSX')])
            if candidates:
                parts_df = []
                for c in candidates:
                    with z.open(c) as f:
                        parts_df.append(pd.read_csv(f, low_memory=False))
                df = pd.concat(parts_df, ignore_index=True)
                print(f"  Loaded {fname} from {zname} ({len(candidates)} parts): {len(df):,} rows")
                return df
    print(f"  WARNING: {fname} not found, skipping")
    return None

parts = []
for fname in ['201811-citibike-tripdata.csv',
              '201812-citibike-tripdata.csv',
              '201901-citibike-tripdata.csv']:
    df_part = _load_month(fname)
    if df_part is not None:
        parts.append(df_part)

df = pd.concat(parts, ignore_index=True)
print(f"Total rows: {len(df):,}")

df['starttime'] = pd.to_datetime(df['starttime'], errors='coerce')
df['stoptime']  = pd.to_datetime(df['stoptime'],  errors='coerce')
df = df.dropna(subset=['starttime'])
df = df[(df['starttime'] >= START) & (df['starttime'] < END)]
print(f"  Rows in date range: {len(df):,}")

st = pd.concat([
    df[['start station id','start station name','start station latitude','start station longitude']]
      .rename(columns={'start station id':'station_id','start station name':'name',
                       'start station latitude':'lat','start station longitude':'lon'}),
    df[['end station id','end station name','end station latitude','end station longitude']]
      .rename(columns={'end station id':'station_id','end station name':'name',
                       'end station latitude':'lat','end station longitude':'lon'}),
], ignore_index=True).dropna()
stations = (st.groupby('station_id')
              .agg({'name':'first','lat':'mean','lon':'mean'})
              .reset_index()
              .sort_values('station_id')
              .reset_index(drop=True))
sid2idx = {sid: i for i, sid in enumerate(stations['station_id'])}
N = len(stations)
print(f"  Stations: {N}")

hours     = pd.date_range(START, END, freq='h', inclusive='left')
h2i       = {h: i for i, h in enumerate(hours)}
T         = len(hours)
print(f"  T={T} hours ({T//24} days)")

df['h_idx']   = df['starttime'].dt.floor('h').map(h2i)
df['s_idx']   = df['start station id'].map(sid2idx)
df['stop_h']  = df['stoptime'].dt.floor('h').map(h2i)
df['e_idx']   = df['end station id'].map(sid2idx)

def build_demand(h_col, z_col):
    valid = df[[h_col, z_col]].dropna()
    mat = np.zeros((T, N), dtype=np.float32)
    hi  = valid[h_col].astype(int).values
    zi  = valid[z_col].astype(int).values
    np.add.at(mat, (hi, zi), 1)
    return mat

pickup  = build_demand('h_idx', 's_idx')
dropoff = build_demand('stop_h', 'e_idx')
demand  = pickup + dropoff
print(f"  demand shape: {demand.shape}, mean={demand.mean():.3f}")

print("Building OD matrix...")
valid_od = df[['h_idx','s_idx','e_idx']].dropna()
hi = valid_od['h_idx'].astype(int).values
si = valid_od['s_idx'].astype(int).values
ei = valid_od['e_idx'].astype(int).values
od = np.zeros((T, N, N), dtype=np.float32)
np.add.at(od, (hi, si, ei), 1)
print(f"  OD shape: {od.shape}, sparsity: {(od>0).mean():.5f}")

np.save(OUT / "bike_hourly_demand.npy", demand)
np.save(OUT / "bike_od_hourly.npy",    od)
stations.to_csv(OUT / "bike_stations.csv", index=False)
pd.DataFrame({'hour': hours.astype(str)}).to_csv(OUT / "bike_hours.csv", index=False)
print(f"\n[OK] NYC Citi Bike 3-month processing done  T={T}  N={N}")
