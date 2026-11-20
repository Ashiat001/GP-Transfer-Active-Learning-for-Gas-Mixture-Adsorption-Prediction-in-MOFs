'''
Create the held-out test sets ONCE, before any training.

For every system, grid (px / pxt) and seed this writes
    Datasets/<system data folder>/splits/<space>/test_idx_seed<N>.csv   grid rows held out for testing
    Datasets/<system data folder>/splits/<space>/pool_seed<N>.csv       remaining rows (full columns) - initial data and AL queries come from here
using the system's default split rule (systems.py, 'split'; 'random' if not given).
With --rule, splits for that rule are written instead (to splits_<rule>/ when it is not the default).

Existing split files are never overwritten (pass --force to regenerate).
Usage: python make_splits.py [--force] [--rule random|lowest_p] [--systems CO2-CH4/irmof1 ...]
'''
import os
import argparse
import numpy as np
import pandas as pd
from systems import SYSTEMS, SEEDS, TEST_FRAC
from heldout import read_grid_file, make_split, split_dir, split_rule, SPLIT_RULES

parser = argparse.ArgumentParser()
parser.add_argument('--force', action = 'store_true')
parser.add_argument('--rule', choices = SPLIT_RULES, default = None)
parser.add_argument('--systems', nargs = '+', default = list(SYSTEMS))
args = parser.parse_args()

for system in args.systems:
    info = SYSTEMS[system]
    rule = split_rule(system, args.rule)
    for space in info['grids']:
        df, spec = read_grid_file(system, space)
        out_dir = split_dir(system, space, rule)
        os.makedirs(out_dir, exist_ok = True)

        for seed in SEEDS:
            test_path = os.path.join(out_dir, f'test_idx_seed{seed}.csv')
            pool_path = os.path.join(out_dir, f'pool_seed{seed}.csv')
            if os.path.exists(test_path) and not args.force:
                print(f'{test_path} exists, skipping')
                continue

            test_rows = make_split(system, space, seed, rule)
            pool_rows = np.setdiff1d(np.arange(len(df)), test_rows)

            pd.DataFrame({'grid_row': test_rows}).to_csv(test_path, index = False)
            df.iloc[pool_rows].to_csv(pool_path, index_label = 'grid_row')
            print(f'{system} {space} seed {seed} ({rule}): {len(test_rows)} test / {len(pool_rows)} pool '
                  f'({100 * TEST_FRAC:.0f}% of {len(df)} held out) -> {out_dir}')
