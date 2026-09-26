"""
TransMod 

    cd /path/to/TransMod
    python data/preprocessing/run_all.py
    python data/preprocessing/run_all.py --city nyc
    python data/preprocessing/run_all.py --city chicago
"""
import sys, time, traceback, argparse
from pathlib import Path
import importlib.util

ROOT = Path(__file__).parent

def run_script(script_path: Path, label: str):
    print(f"\n{'='*60}")
    print(f"    {label}")
    print(f"{'='*60}")
    t0 = time.time()
    spec = importlib.util.spec_from_file_location("module", script_path)
    mod  = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        elapsed = time.time() - t0
        print(f"\n  [OK]  ({elapsed:.1f}s)")
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"\n  [FAIL]  ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return False

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--city', choices=['nyc', 'chicago', 'all'], default='all')
    p.add_argument('--skip', nargs='*', default=[],
                   help=': bike metro ridehailing')
    args = p.parse_args()

    scripts_nyc = [
        (ROOT / "process_nyc_bike.py",         "NYC Citi Bike (Jan 2018)"),
        (ROOT / "process_nyc_metro.py",         "NYC MTA  (Jan 2018)"),
        (ROOT / "process_nyc_ridehailing.py",   "NYC TLC  (Jan 2018)"),
    ]
    scripts_chi = [
        (ROOT / "process_chicago_bike.py",      "Chicago Divvy  (Jan 2018)"),
        (ROOT / "process_chicago_metro.py",     "Chicago CTA  (Jan 2018)"),
        (ROOT / "process_chicago_ridehailing.py","Chicago TNP  (Nov 2018)"),
    ]

    if args.city in ('nyc', 'all'):
        scripts = scripts_nyc
    else:
        scripts = []
    if args.city in ('chicago', 'all'):
        scripts = scripts + scripts_chi

    skip_kw = [s.lower() for s in args.skip]
    scripts = [(p, l) for p, l in scripts
               if not any(kw in l.lower() for kw in skip_kw)]

    results = {}
    total_t0 = time.time()
    for script, label in scripts:
        results[label] = run_script(script, label)

    print(f"\n{'='*60}")
    print("  ")
    print(f"{'='*60}")
    for label, ok in results.items():
        status = "[OK]" if ok else "[FAIL]"
        print(f"  {status}  {label}")
    total = time.time() - total_t0
    ok_n  = sum(results.values())
    print(f"\n  {ok_n}/{len(results)}   : {total:.1f}s")

    print(f"\n :")
    for city in ['nyc', 'chicago']:
        out_dir = ROOT.parent / "processed" / city
        if out_dir.exists():
            print(f"\n  {city}/")
            for f in sorted(out_dir.iterdir()):
                size = f.stat().st_size / 1024**2
                print(f"    {f.name:45s} {size:7.1f} MB")

if __name__ == '__main__':
    main()
