"""Extract all AIOps 2020 daily ZIP files using 7-zip."""
import subprocess, os, sys

inner_dir = r'C:\Users\Sainath S Borule\Downloads\microservice-anomaly-detection\data\outer'
data_dir  = r'C:\Users\Sainath S Borule\Downloads\microservice-anomaly-detection\data'
password  = '\u8fd9\u662f\u6311\u6218\u8d5b\u521d\u8d5b\u7b54\u6848\u5bc6\u7801'
seven_zip = r'C:\Program Files\7-Zip\7z.exe'

sub = [d for d in os.listdir(inner_dir) if os.path.isdir(os.path.join(inner_dir, d))][0]
aiops_dir = os.path.join(inner_dir, sub)
zips = sorted([f for f in os.listdir(aiops_dir) if f.startswith('2020_') and f.endswith('.zip')])
print(f'Found {len(zips)} day ZIPs to extract...')

for z in zips:
    day = z.replace('.zip', '')
    dest = os.path.join(data_dir, day)
    if os.path.exists(dest) and len(os.listdir(dest)) > 0:
        print(f'  SKIP {day} (already extracted)')
        continue
    os.makedirs(dest, exist_ok=True)
    zip_path = os.path.join(aiops_dir, z)
    print(f'  Extracting {day}...', end='', flush=True)
    r = subprocess.run(
        [seven_zip, 'x', zip_path, f'-o{dest}', f'-p{password}', '-y'],
        capture_output=True, text=True, encoding='utf-8', errors='replace'
    )
    if 'Everything is Ok' in r.stdout:
        print(' OK')
    else:
        print(f' FAIL (rc={r.returncode})')
        print(r.stdout[-300:])

print('All done!')
