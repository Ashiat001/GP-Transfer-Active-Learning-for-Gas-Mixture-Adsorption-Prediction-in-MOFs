'''
Paired summary of an ablation folder (heldout.run_ablation): every arm is trained on the same points for each
split seed and size, so the GP-with-pure-mean arm ('TGP_xP') is compared seed by seed with the
ordinary GP ('GP').

One row per run (gas), space, size and arm: the number of seeds, the median and worst test R2, the median
MAE / MAPE and, for the TGP arms, the paired difference arm - GP in R2 per seed (median) and in how many seeds
the arm is better / tied / worse than GP (an R2 difference of at most TIE counts as a tie).
Size 0 is the pure-component mean alone (TGP arms only, so no paired columns).

Usage: python ablation_summary.py <ablation folder>     e.g. python job_script/ablation_summary.py CO2-CH4/cubtc/Ablation_Test
Writes <ablation folder>/Ablation_summary.csv. collect_metrics.py gives the plain mean / std per arm and size.
'''
import os
import sys
import glob
import pandas as pd

TIE = 0.005     # R2 differences up to this size count as a tie

folder = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else '.')

frames = []
for path in glob.glob(os.path.join(folder, '*', 'Metrics_*_seed*.csv')):
    df = pd.read_csv(path)
    df['run'] = os.path.basename(os.path.dirname(path))
    frames.append(df)
if not frames:
    sys.exit(f'No Metrics_*_seed*.csv files found under {folder}')
m = pd.concat(frames, ignore_index = True)

keys = ['run', 'space', 'n_requested', 'arm']
g = m.groupby(keys)
summary = pd.DataFrame({'seeds': g['seed'].nunique(), 'R2_median': g['R2'].median(), 'R2_worst': g['R2'].min(),
                        'MAE_median': g['MAE'].median(), 'MAPE_median': g['MAPE'].median()})

# paired comparison with the ordinary GP (same run, space, size and seed)
gp = m[m['arm'] == 'GP'].set_index(['run', 'space', 'n_requested', 'seed'])['R2']
tgp = m[m['arm'] != 'GP'].set_index(['run', 'space', 'n_requested', 'seed', 'arm'])['R2']
if len(gp) and len(tgp):
    diff = (tgp - gp.reindex(tgp.index.droplevel('arm')).values).dropna()
    d = diff.groupby(level = keys)
    summary = summary.join(pd.DataFrame({'dR2_vs_GP_median': d.median(),
                                         'better_than_GP': d.apply(lambda v: int((v > TIE).sum())),
                                         'tied_with_GP': d.apply(lambda v: int((v.abs() <= TIE).sum())),
                                         'worse_than_GP': d.apply(lambda v: int((v < -TIE).sum()))}))

summary = summary.sort_index()
summary.to_csv(os.path.join(folder, 'Ablation_summary.csv'))
print(summary.round(4).to_string())
