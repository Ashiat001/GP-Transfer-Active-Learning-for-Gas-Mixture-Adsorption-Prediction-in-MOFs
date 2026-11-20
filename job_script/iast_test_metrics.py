'''IAST scored on the same held-out test points as TAL, with the same metric formulas.

For every system / space / gas and split seed:
  - the test points are the rows marked split == 'test' in the TAL run's Pred_<run>_<space>_seed<N>.csv
    (so BI points removed from the TAL test set are removed here too, and IRMOF / NU x = 0 rows are absent)
  - IAST and TAL are scored on those points with the formulas of heldout.train_and_evaluate:
    R2 and MAE against the GCMC uptake (zero set to 1e-4), MAPE with the system's rule
    ('all': every test point, zero set to 1e-4; 'nonzero': only GCMC uptake > 0)
  - the TAL numbers are checked against its Metrics file (they should be identical)
IAST on the full grid is also given (same formulas), for reference.

Needs the TAL runs first: CO2-CH4/<mof>/AF_Test/TAL_Pure_*_M and CO2-H2S, Xe-Kr AL_TAL/TAL_Pure_* (all 5 seeds,
job_script/run_all.sh), since it reads their Pred_ and Metrics_ files. IAST predictions: IAST_Calculation/.
Usage: cd IAST_Calculation && python ../job_script/iast_test_metrics.py

Read only. Writes to OUT:
  IAST_test_metrics_seeds.csv     one row per system / space / gas / method / seed
  IAST_test_metrics_summary.csv   mean and standard deviation (ddof 1, as collect_metrics) over the 5 seeds
'''
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))     # repository folder
OUT = os.getcwd()                                                    # results go to the folder you run it from
SEEDS = [0, 1, 2, 3, 4]
ZERO_UPTAKE = 0.0001        # as heldout.ZERO_UPTAKE

# P-X-T IAST uses the mixture GCMC at the largest mole fraction of each gas (0.98) as that gas's pure isotherm.
# True: leave those rows out of the test points (for IAST and TAL alike). False: keep the TAL test set as it is.
DROP_PXT_IAST_INPUT = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from systems import SYSTEMS          # MAPE rule per system ('mape', default 'all') and data folder ('dir')

CO_CH = {'CO2': 'AF_Test/TAL_Pure_CO_CO_M', 'CH4': 'AF_Test/TAL_Pure_CH_CH_M'}
RUNS = {    # system: (spaces, {gas: TAL run folder})
    'CO2-CH4/cubtc':   (['px', 'pxt'], CO_CH),
    'CO2-CH4/irmof1':  (['px'], CO_CH),
    'CO2-CH4/irmof10': (['px'], CO_CH),
    'CO2-CH4/nu800':   (['px'], CO_CH),
    'CO2-H2S':         (['px', 'pxt'], {'CO2': 'AL_TAL/TAL_Pure_CO_CO', 'H2S': 'AL_TAL/TAL_Pure_HS_HS'}),
    'Xe-Kr':           (['px', 'pxt'], {'Kr': 'AL_TAL/TAL_Pure_Kr_Kr', 'Xe': 'AL_TAL/TAL_Pure_Xe_Xe'}),
}


def iast_dir(system):
    '''IAST_Calculation/ has the same layout as Datasets/ (e.g. Datasets/IRMOF-1 -> IAST_Calculation/IRMOF-1)'''
    return os.path.join(ROOT, SYSTEMS[system]['dir'].replace('Datasets', 'IAST_Calculation', 1))


def clean(y):
    y = np.asarray(y, dtype = float)
    return np.where(y <= 0, ZERO_UPTAKE, y)


def score(system, gcmc, pred):
    '''R2, MAE, MAPE (fraction) as in heldout.train_and_evaluate'''
    gcmc, pred = np.asarray(gcmc, dtype = float), np.asarray(pred, dtype = float)
    y = clean(gcmc)
    keep = gcmc > 0 if SYSTEMS[system].get('mape', 'all') == 'nonzero' else np.ones(len(gcmc), dtype = bool)
    return {'n_test': len(y),
            'R2': 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2),
            'MAE': np.mean(np.abs(y - pred)),
            'MAPE': np.mean(np.abs((pred[keep] - clean(gcmc[keep])) / clean(gcmc[keep]))),
            'n_test_MAPE': int(keep.sum())}


def read_iast(system, space, gas):
    '''IAST file in standard columns P, (T), x, GCMC, IAST'''
    d = pd.read_csv(os.path.join(iast_dir(system), f'iast_pred_{space}.csv'))
    pcol = 'Pressure(Pa)' if 'Pressure(Pa)' in d else 'Pressure [Pa]'
    xcol = [c for c in d.columns if c.lower() in (f'{gas}_mole_frac'.lower(), f'mole fraction {gas}'.lower())][0]
    out = pd.DataFrame({'P': d[pcol].values, 'x': d[xcol].values,
                        'GCMC_iast_file': d[f'GCMC_{gas}_uptake'].values, 'IAST': d[f'IAST_{gas}_uptake'].values})
    if space == 'pxt':
        out['T'] = d['Temperature(K)'].values
    return out


def keys(df, space):
    '''match on pressure (to 1 Pa), temperature and mole fraction (4 decimals)'''
    cols = [df['P'].round(0), df['x'].round(4)] + ([df['T'].round(1)] if space == 'pxt' else [])
    return pd.MultiIndex.from_arrays(cols)


rows = []
for system, (spaces, runs) in RUNS.items():
    for space in spaces:
        for gas, run in runs.items():
            iast = read_iast(system, space, gas)
            iast.index = keys(iast, space)
            assert iast.index.is_unique, (system, space, gas, 'IAST rows not unique')
            name = os.path.basename(run)

            # IAST on the full grid (reference)
            ok = iast['IAST'].notna()
            rows.append({'system': system, 'space': space, 'gas': gas, 'method': 'IAST_full_grid', 'seed': np.nan,
                         'n_IAST_missing': int((~ok).sum()), **score(system, iast.loc[ok, 'GCMC_iast_file'], iast.loc[ok, 'IAST'])})

            for seed in SEEDS:
                pred = pd.read_csv(os.path.join(ROOT, system, run, f'Pred_{name}_{space}_seed{seed}.csv'))
                test = pred[pred['split'] == 'test'].rename(columns = {'Pressure(Pa)': 'P', 'Temperature(K)': 'T',
                                                                       'mole fraction': 'x'})
                tal_all = score(system, test['GCMC'], test['Predicted'])

                # check: same numbers as the TAL Metrics file
                met = pd.read_csv(os.path.join(ROOT, system, run, f'Metrics_{name}_seed{seed}.csv'))
                met = met[met['space'] == space].iloc[0]
                for k in ['n_test', 'R2', 'MAE', 'MAPE']:
                    if not np.isclose(tal_all[k], met[k], rtol = 1e-9, atol = 1e-12):
                        print(f'  CHECK {system} {space} {gas} seed {seed}: TAL {k} {tal_all[k]} vs Metrics file {met[k]}')

                if space == 'pxt' and DROP_PXT_IAST_INPUT:
                    test = test[test['x'].round(4) < round(iast['x'].max(), 4)]

                # IAST at the same points
                test.index = keys(test, space)
                missing = test.index.difference(iast.index)
                assert len(missing) == 0, (system, space, gas, seed, f'{len(missing)} test points not in the IAST file')
                both = test.join(iast[['GCMC_iast_file', 'IAST']])
                bad = ~np.isclose(both['GCMC'], both['GCMC_iast_file'], rtol = 1e-6, atol = 1e-9)
                assert not bad.any(), (system, space, gas, seed, f'{bad.sum()} points with a different GCMC uptake')
                ok = both['IAST'].notna()      # points where IAST gave no value are left out of the IAST score

                info = {'system': system, 'space': space, 'gas': gas, 'seed': seed}
                rows.append({**info, 'method': 'IAST', 'n_IAST_missing': int((~ok).sum()),
                             **score(system, both.loc[ok, 'GCMC'], both.loc[ok, 'IAST'])})
                rows.append({**info, 'method': 'TAL', 'n_IAST_missing': 0, 'run': run,
                             **score(system, both['GCMC'], both['Predicted'])})

seeds = pd.DataFrame(rows)
seeds.to_csv(os.path.join(OUT, 'IAST_test_metrics_seeds.csv'), index = False)

g = seeds.groupby(['system', 'space', 'gas', 'method'], sort = False)
summary = g[['n_test', 'n_IAST_missing']].mean()
for k in ['R2', 'MAE', 'MAPE']:
    summary[f'{k}_mean'] = g[k].mean()
    summary[f'{k}_std'] = g[k].std()     # NaN for the single full-grid value
summary['n_seeds'] = g['R2'].count()
summary = summary.reset_index()
summary.to_csv(os.path.join(OUT, 'IAST_test_metrics_summary.csv'), index = False)

pd.set_option('display.width', 200)
show = summary.copy()
show['MAPE (%)'] = (100 * show['MAPE_mean']).round(2).astype(str) + ' ± ' + (100 * show['MAPE_std']).round(2).astype(str)
show['R2'] = show['R2_mean'].round(4).astype(str) + ' ± ' + show['R2_std'].round(4).astype(str)
show['MAE (mg/g)'] = show['MAE_mean'].round(2).astype(str) + ' ± ' + show['MAE_std'].round(2).astype(str)
print(show[['system', 'space', 'gas', 'method', 'n_test', 'R2', 'MAE (mg/g)', 'MAPE (%)']].to_string(index = False))
print('\nwritten', os.path.join(OUT, 'IAST_test_metrics_seeds.csv'), 'and', os.path.join(OUT, 'IAST_test_metrics_summary.csv'))
