"""
NYC MTA  (Nov 2018 - Jan 2019, 3 months)
  data/nyc/metro/turnstile_181*.txt   (Nov+Dec 2018)
  data/nyc/metro/turnstile_190*.txt   (Jan 2019)
  data/nyc/metro/stops.txt
  processed/nyc/metro_hourly_entries.npy  [T=2208, N]
  processed/nyc/metro_hourly_exits.npy    [T=2208, N]
  processed/nyc/metro_stations.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

ROOT = Path(__file__).parent.parent
RAW  = ROOT / "nyc/metro"
OUT  = ROOT / "processed/nyc"
OUT.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp('2018-11-01')
END   = pd.Timestamp('2019-02-01')   # exclusive, 92 days = 2208 hours

print(" GTFS ...")
gtfs = pd.read_csv(RAW / "stops.txt")
parent = gtfs[gtfs['location_type'] == 1][['stop_name', 'stop_lat', 'stop_lon']].copy()
parent = parent.rename(columns={'stop_name': 'name', 'stop_lat': 'lat', 'stop_lon': 'lon'})
parent = parent.drop_duplicates('name').reset_index(drop=True)
parent['name_upper'] = parent['name'].str.upper()
print(f"  GTFS parent stations: {len(parent)}")

files = sorted(glob(str(RAW / "turnstile_181*.txt"))) + \
        sorted(glob(str(RAW / "turnstile_190*.txt")))
print(f"  Turnstile : {len(files)}")

dfs = []
for f in files:
    try:
        chunk = pd.read_csv(f, header=0, skipinitialspace=True, low_memory=False)
        chunk.columns = chunk.columns.str.strip()
        dfs.append(chunk)
    except Exception as e:
        print(f"   {Path(f).name}: {e}")

df = pd.concat(dfs, ignore_index=True)
print(f"  : {len(df):,}")

col_map = {
    'C/A': 'ca', 'UNIT': 'unit', 'SCP': 'scp', 'STATION': 'station',
    'DATE': 'date', 'TIME': 'time', 'ENTRIES': 'entries', 'EXITS': 'exits'
}
df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
df['station'] = df['station'].str.strip()
df['entries'] = pd.to_numeric(df['entries'].astype(str).str.strip(), errors='coerce')
df['exits']   = pd.to_numeric(df['exits'].astype(str).str.strip(),   errors='coerce')

df['datetime'] = pd.to_datetime(
    df['date'].astype(str).str.strip() + ' ' + df['time'].astype(str).str.strip(),
    format='%m/%d/%Y %H:%M:%S', errors='coerce')
df = df.dropna(subset=['datetime', 'entries', 'exits'])
df = df[(df['datetime'] >= START) & (df['datetime'] < END)]
print(f"  3-month : {len(df):,}")

df = df.sort_values(['ca', 'unit', 'scp', 'datetime'])
df['entry_diff'] = df.groupby(['ca', 'unit', 'scp'])['entries'].diff()
df['exit_diff']  = df.groupby(['ca', 'unit', 'scp'])['exits'].diff()

THRESHOLD = 50_000
df = df[(df['entry_diff'] >= 0) & (df['entry_diff'] < THRESHOLD)]
df = df[(df['exit_diff']  >= 0) & (df['exit_diff']  < THRESHOLD)]

df['hour'] = df['datetime'].dt.floor('h')
agg = df.groupby(['station', 'hour']).agg(
    entries=('entry_diff', 'sum'),
    exits=('exit_diff', 'sum')
).reset_index()

station_names = sorted(agg['station'].unique())
station_upper = [s.upper() for s in station_names]

# For each turnstile station, find the first GTFS parent whose name contains it
# or is contained by it
def match_coord(sname_up):
    # exact or substring match
    mask = parent['name_upper'].str.contains(sname_up, regex=False, na=False) | \
           pd.Series([sname_up], dtype=str).str.contains(parent['name_upper'].iloc[0] if len(parent) else '', regex=False, na=False).iloc[0]
    # simpler: check if sname substring of gtfs OR gtfs substring of sname
    for _, row in parent.iterrows():
        if sname_up in row['name_upper'] or row['name_upper'] in sname_up:
            return row['lat'], row['lon']
    return None, None

station_info = []
for sname, sup in zip(station_names, station_upper):
    lat, lon = match_coord(sup)
    station_info.append({'name': sname, 'lat': lat, 'lon': lon})

station_df = pd.DataFrame(station_info).dropna(subset=['lat', 'lon']).reset_index(drop=True)
station_to_idx = {n: i for i, n in enumerate(station_df['name'])}
N = len(station_df)
print(f"  : {N}")

hours = pd.date_range(START, END, freq='h', inclusive='left')
hour_to_idx = {h: i for i, h in enumerate(hours)}
T = len(hours)
print(f"   T={T}")

agg['s_idx'] = agg['station'].map(station_to_idx)
agg['h_idx'] = agg['hour'].map(hour_to_idx)
agg_valid = agg.dropna(subset=['s_idx', 'h_idx']).copy()
agg_valid['s_idx'] = agg_valid['s_idx'].astype(int)
agg_valid['h_idx'] = agg_valid['h_idx'].astype(int)

entries_mat = np.zeros((T, N), dtype=np.float32)
exits_mat   = np.zeros((T, N), dtype=np.float32)

np.add.at(entries_mat, (agg_valid['h_idx'].values, agg_valid['s_idx'].values),
           agg_valid['entries'].values.astype(np.float32))
np.add.at(exits_mat,   (agg_valid['h_idx'].values, agg_valid['s_idx'].values),
           agg_valid['exits'].values.astype(np.float32))

print(f"  entries shape: {entries_mat.shape}, : {entries_mat.mean():.2f}")
print(f"  exits   shape: {exits_mat.shape},   : {exits_mat.mean():.2f}")

np.save(OUT / "metro_hourly_entries.npy", entries_mat)
np.save(OUT / "metro_hourly_exits.npy",   exits_mat)
station_df.to_csv(OUT / "metro_stations.csv", index=False)

print(f"\n[OK] NYC MTA 3-month   T={T}  N={N}")
print(f"   entries: {entries_mat.shape} -> {OUT}/metro_hourly_entries.npy")
