'''IAST R2 in Cu-BTC P-X on the held-out TEST points: all test points vs test points at P <= 10 bar.
Same test points, IAST files and R2 formula as iast_test_metrics.py (Table S5):
  test points = rows with split == 'test' in the TAL run's Pred_<run>_px_seed<N>.csv
  GCMC uptake <= 0 set to 1e-4; points where IAST gave no value are left out.
Needs the TAL runs first (CO2-CH4/cubtc/AF_Test/TAL_Pure_*_M and Xe-Kr/AL_TAL/TAL_Pure_*, all 5 seeds).
Usage: cd IAST_Calculation && python ../job_script/iast_10bar_test.py
Read only. Prints the per-seed values and mean +- std (ddof 1, as collect_metrics) over the 5 seeds, and writes
to the folder you run it from:
  iast_10bar_test_points.csv   every test point used: system, gas, seed, P, x, GCMC, IAST, residual^2, le_10bar
  iast_10bar_test_seeds.csv    n, SS_res, SS_tot, R2 per system / gas / seed / pressure range
'''
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository folder
SEEDS = [0, 1, 2, 3, 4]
ZERO_UPTAKE = 0.0001
P_MAX = 1e6 + 1     # 10 bar in Pa
RUNS = {'CO2-CH4/cubtc': ('IAST_Calculation/Cu-BTC/CO2-CH4', {'CO2': 'AF_Test/TAL_Pure_CO_CO_M', 'CH4': 'AF_Test/TAL_Pure_CH_CH_M'}),
        'Xe-Kr':         ('IAST_Calculation/Cu-BTC/Xe-Kr',   {'Xe': 'AL_TAL/TAL_Pure_Xe_Xe', 'Kr': 'AL_TAL/TAL_Pure_Kr_Kr'})}

points, rows = [], []
for system, (iast_dir, runs) in RUNS.items():
    d = pd.read_csv(os.path.join(ROOT, iast_dir, 'iast_pred_px.csv'))
    for gas, run in runs.items():
        iast = pd.DataFrame({'P': d['Pressure(Pa)'], 'x': d[f'{gas}_mole_frac'], 'IAST': d[f'IAST_{gas}_uptake']})
        iast.index = pd.MultiIndex.from_arrays([iast['P'].round(0), iast['x'].round(4)])
        name = os.path.basename(run)
        for seed in SEEDS:
            pred = pd.read_csv(os.path.join(ROOT, system, run, f'Pred_{name}_px_seed{seed}.csv'))
            test = pred[pred['split'] == 'test'].rename(columns = {'Pressure(Pa)': 'P', 'mole fraction': 'x'})
            test.index = pd.MultiIndex.from_arrays([test['P'].round(0), test['x'].round(4)])
            t = test[['P', 'x', 'GCMC']].join(iast[['IAST']])
            t = t[t['IAST'].notna()].reset_index(drop = True)
            t['GCMC'] = np.where(t['GCMC'] <= 0, ZERO_UPTAKE, t['GCMC'])
            t['resid2'] = (t['GCMC'] - t['IAST']) ** 2
            t['le_10bar'] = t['P'] <= P_MAX
            points.append(t.assign(system = system, gas = gas, seed = seed))
            for label, m in [('all test', np.ones(len(t), bool)), ('test P<=10 bar', t['le_10bar'].values)]:
                y, yhat = t.loc[m, 'GCMC'], t.loc[m, 'IAST']
                ss_res, ss_tot = ((y - yhat) ** 2).sum(), ((y - y.mean()) ** 2).sum()
                rows.append({'system': system, 'gas': gas, 'range': label, 'seed': seed, 'n': int(m.sum()),
                             'SS_res': ss_res, 'SS_tot': ss_tot, 'R2': 1 - ss_res / ss_tot})

pts = pd.concat(points)[['system', 'gas', 'seed', 'P', 'x', 'GCMC', 'IAST', 'resid2', 'le_10bar']]
pts.to_csv('iast_10bar_test_points.csv', index = False)
seeds = pd.DataFrame(rows)
seeds.to_csv('iast_10bar_test_seeds.csv', index = False)

pd.set_option('display.width', 200)
print(seeds.pivot_table(index = ['system', 'gas', 'range'], columns = 'seed', values = 'R2', sort = False).round(3))
g = seeds.groupby(['system', 'gas', 'range'], sort = False)
summ = pd.DataFrame({'n_mean': g['n'].mean(), 'n_min': g['n'].min(), 'n_max': g['n'].max(),
                     'R2_mean': g['R2'].mean(), 'R2_std': g['R2'].std()})
print('\n', summ.round(3).to_string())
print('\nwrote iast_10bar_test_points.csv and iast_10bar_test_seeds.csv')
