'''
Collect the per-seed test-set metrics of one analysis folder into a mean +/- std summary.

Usage: python collect_metrics.py <analysis folder>     e.g. CO2-CH4/cubtc/Data_Distribution
Writes <analysis folder>/Summary_metrics.csv (and All_metrics.csv with every seed).
'''
import os
import re
import sys
import glob
import pandas as pd

folder = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')

frames = []
for path in glob.glob(os.path.join(folder, '*', 'Metrics_*_seed*.csv')):
    df = pd.read_csv(path)
    df['run'] = os.path.basename(os.path.dirname(path))
    df['file'] = re.sub(r'_seed\d+\.csv$', '', os.path.basename(path))
    frames.append(df)

if not frames:
    sys.exit(f'No Metrics_*_seed*.csv files found under {folder}')

all_metrics = pd.concat(frames, ignore_index = True)
all_metrics.to_csv(os.path.join(folder, 'All_metrics.csv'), index = False)

# 'arm' (GP / TGP) and 'n_requested' only exist in ablation / data-size results
keys = ['run', 'file', 'space'] + [c for c in ['arm', 'n_requested'] if c in all_metrics.columns]
values = [c for c in ['R2', 'MAE', 'MAPE', 'n_train_final', 'n_iters', 'n_test', 'n_test_MAPE'] if c in all_metrics.columns]
summary = all_metrics.groupby(keys, dropna = False)[values].agg(['mean', 'std'])
summary['n_seeds'] = all_metrics.groupby(keys, dropna = False)['seed'].nunique()
summary.columns = ['_'.join(c) if isinstance(c, tuple) and c[1] else c[0] if isinstance(c, tuple) else c for c in summary.columns]
summary.to_csv(os.path.join(folder, 'Summary_metrics.csv'))
print(summary.to_string())
