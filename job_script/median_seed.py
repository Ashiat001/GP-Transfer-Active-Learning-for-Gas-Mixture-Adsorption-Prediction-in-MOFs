'''
Pick the split seed to show in the prediction plots (Plot/), one seed per system and space (PX, PXT),
used in all of its plots:
  isotherm / residual plots   TAL M (PAC stop), both gases
  AL vs TAL plots             AL M and TAL M (PAC stop), both gases: sampled points and GP MRE per iteration
The seed is chosen over the four runs (AL and TAL of both gases). Same split seed = same test set and the same
initial points for AL and TAL.

Runs: CO2-CH4/<mof>/AF_Test (AL_<g>_M, AL_<g>_pxt, TAL_Pure_<g>_<g>_M), CO2-H2S and Xe-Kr AL_TAL
(AL_<g>, AL_<g>_pxt, TAL_Pure_<g>_<g>).

Rule: within each run and space the 5 seeds are ranked by test R2, 1 = best, 3 = median. The seed shown is the
one with the smallest total |rank - 3| over the four runs (ties: smallest total |R2 - median R2|, then the lower
seed). If every run has the same median seed, that seed is shown.
Seeds chosen by hand instead of by the rule are listed in CHOSEN (column 'chosen_by' says which).

Usage: python median_seed.py [system ...] [--out DIR]     e.g. python median_seed.py Xe-Kr CO2-CH4/cubtc
       (default: every system; Median_seed.csv is written in Plot/, or in DIR)
'''
import os
import ast
import sys
import argparse
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEEDS = [0, 1, 2, 3, 4]

# system: (analysis folder, gases as (run tag, gas name), spaces, AL run per space, TAL run)
CO2_CH4 = ('AF_Test', [('CO', 'CO2'), ('CH', 'CH4')], {'px': 'AL_{g}_M', 'pxt': 'AL_{g}_pxt'}, 'TAL_Pure_{g}_{g}_M')
SYSTEMS = {
    'CO2-CH4/cubtc':   CO2_CH4 + (['px', 'pxt'],),
    'CO2-CH4/irmof1':  CO2_CH4 + (['px'],),
    'CO2-CH4/irmof10': CO2_CH4 + (['px'],),
    'CO2-CH4/nu800':   CO2_CH4 + (['px'],),
    'CO2-H2S': ('AL_TAL', [('CO', 'CO2'), ('HS', 'H2S')], {'px': 'AL_{g}', 'pxt': 'AL_{g}_pxt'}, 'TAL_Pure_{g}_{g}', ['px', 'pxt']),
    'Xe-Kr':   ('AL_TAL', [('Kr', 'Kr'), ('Xe', 'Xe')], {'px': 'AL_{g}', 'pxt': 'AL_{g}_pxt'}, 'TAL_Pure_{g}_{g}', ['px', 'pxt']),
}

# (system, space): (seed, reason) chosen by hand instead of the rule
# Empty since 2026-11-11: the IRMOF-1 PX hand pick (seed 0) was made on the runs with the x = 0 rows; on the re-runs
# without them the user chose the rule's seed everywhere.
CHOSEN = {}

parser = argparse.ArgumentParser()
parser.add_argument('systems', nargs = '*', default = list(SYSTEMS))
parser.add_argument('--out', help = 'folder for Median_seed.csv (default: Plot/)')
args = parser.parse_args()


def check_cfg(script):
    '''the plotted runs must be M (LHS start) with the PAC stop'''
    for node in ast.parse(open(script).read()).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'cfg':
            cfg = ast.literal_eval(node.value)
            if cfg.get('init') != 'lhs' or cfg.get('stop_rule') != 'pac':
                sys.exit(f'{script}: init {cfg.get("init")}, stop {cfg.get("stop_rule")} (expected lhs, pac)')


def test_r2(folder, run, space):
    '''test R2 per seed (NaN where the seed has not finished)'''
    check_cfg(os.path.join(folder, f'{run}.py'))
    r2 = {}
    for seed in SEEDS:
        path = os.path.join(folder, run, f'Metrics_{run}_seed{seed}.csv')
        if os.path.exists(path):
            m = pd.read_csv(path)
            m = m[m['space'] == space]
            if len(m):
                r2[seed] = m['R2'].iloc[0]
    return pd.Series(r2, index = SEEDS, dtype = float)


def pick(table):
    '''table: seeds x runs of test R2 (all 5 seeds finished). Returns the seed shown and the ranks.'''
    rank = table.rank(ascending = False, method = 'first')    # 1 = best R2
    off_rank = (rank - 3).abs().sum(axis = 1)
    off_r2 = (table - table.median()).abs().sum(axis = 1)
    order = pd.DataFrame({'off_rank': off_rank.values, 'off_r2': off_r2.values, 'seed': table.index.values})
    return int(order.sort_values(['off_rank', 'off_r2', 'seed'])['seed'].iloc[0]), rank


rows = []
for system in args.systems:
    if system not in SYSTEMS:
        sys.exit(f'Unknown system {system}; choose from {list(SYSTEMS)}')
    analysis, gases, al_name, tal_name, spaces = SYSTEMS[system]
    folder = os.path.join(ROOT, system, analysis)
    for space in spaces:
        runs = {}    # run -> (method, gas)
        for tag, gas in gases:
            runs[tal_name.format(g = tag)] = ('TAL', gas)
        for tag, gas in gases:
            runs[al_name[space].format(g = tag)] = ('AL', gas)
        table = pd.DataFrame({run: test_r2(folder, run, space) for run in runs})
        unfinished = sorted(c for c in table if table[c].isna().any())
        if unfinished:
            print(f'{system:16s} {space:4s} not finished: {unfinished}')
            continue
        seed, rank = pick(table)
        chosen_by = 'rule'
        if (system, space) in CHOSEN:
            rule_seed = seed
            seed, reason = CHOSEN[(system, space)]
            chosen_by = f'by hand (rule seed {rule_seed}): {reason}'
        for run in table:
            method, gas = runs[run]
            row = {'system': system, 'space': space, 'seed_shown': seed, 'chosen_by': chosen_by, 'method': method, 'gas': gas,
                   'run': f'{analysis}/{run}', 'rank_shown': int(rank.loc[seed, run]), 'R2_shown': table.loc[seed, run],
                   'R2_median': table[run].median(), 'own_median_seed': int(rank.index[rank[run] == 3][0])}
            row.update({f'R2_seed{s}': table.loc[s, run] for s in SEEDS})
            rows.append(row)

out = pd.DataFrame(rows)
target = os.path.abspath(args.out) if args.out else os.path.join(ROOT, 'Plot')
os.makedirs(target, exist_ok = True)
out.to_csv(os.path.join(target, 'Median_seed.csv'), index = False)

if len(out):
    for (system, space), fig in out.groupby(['system', 'space'], sort = False):
        ranks = ', '.join(f'{m} {g} {k}' for m, g, k in zip(fig['method'], fig['gas'], fig['rank_shown']))
        by_hand = '   by hand' if fig['chosen_by'].iloc[0] != 'rule' else ''
        print(f'{system:16s} {space:4s} seed {fig["seed_shown"].iloc[0]}   (rank: {ranks}){by_hand}')
print(f'\nWritten {os.path.join(target, "Median_seed.csv")}')
