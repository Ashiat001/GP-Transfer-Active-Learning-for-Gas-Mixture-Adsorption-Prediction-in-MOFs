'''
Held-out version of the AL / TAL workflow.

The GCMC grid of every system is split ONCE per seed into
    - test set (20 %, stratified): never used for training, querying, scaling or stopping
    - pool (80 %): initial training points and every AL query come from here
Input scaling statistics are computed from the pool only; uptakes are log10 transformed
(no data-derived statistics). Metrics are computed on the test set only, while
predictions are still written for the full GCMC grid (with a 'split' column) for plotting.

The GP model, kernel, acquisition (max sigma/|mu|) and stopping rules are unchanged
from the original scripts (see GPR.py). In TAL the pure-component GP now enters the mixture GP
at the partial pressure x P, scaled as the pure GP was trained (prior 'xP', see prior_mean).
In IRMOF-1, IRMOF-10 and NU-800 the grid rows where the modelled gas is absent (x = 0) are left out (see x0_rule).
'''
import os
import ast
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import gpflow
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import GPR
from systems import SYSTEMS, PURE_P, PURE_Y, TEST_FRAC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ZERO_UPTAKE = 0.0001    # zero / negative uptakes are replaced with this value (as before)
PAC_TOL = 0.02          # sigma/|mu| threshold used to count a point as "confident" for PAC
LHS_SEED = 40           # seed used by train_data_generator in the original scripts
MAX_ITER_MIX = 2000
MAX_ITER_PURE = 500

gpConfig = {'kernel':'RBF',
            'useWhiteKernel':True,
            'trainLikelihood':True}

# ===================== data loading =====================

def system_dir(system):
    '''Data folder of the system in Datasets/ (systems.py).'''
    return os.path.join(ROOT, SYSTEMS[system]['dir'])

def read_grid_file(system, space):
    spec = SYSTEMS[system]['grids'][space]
    path = os.path.join(system_dir(system), spec['file'])
    if path.endswith('.xlsx'):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    return df, spec

def load_grid(system, space, gas):
    '''Return the GCMC grid for one gas with standard columns P [Pa], (T [K]), x, y.
    The index is the row number in the original grid file. For systems with 'drop_absent' (see x0_rule)
    the rows where this gas is absent (x = 0) are left out, so the index then has gaps.'''
    df, spec = read_grid_file(system, space)
    grid = pd.DataFrame({'P': df[spec['P']].values})
    if spec['T'] is not None:
        grid['T'] = df[spec['T']].values
    grid['x'] = df[spec['x'][gas]].values
    grid['y'] = df[spec['y'][gas]].values
    if grid.isna().any().any():
        raise ValueError(f'NaN values found in grid {spec["file"]}')
    if x0_rule(system) == 'dropped':
        grid = grid[grid['x'] > 0]
    return grid

def x0_rule(system):
    '''Grid rows where the modelled gas is absent (x = 0): 'dropped' for systems with 'drop_absent' in systems.py
    (IRMOF-1, IRMOF-10, NU-800), else 'kept' (the Cu-BTC grids have no such rows). Their uptake is 0 by definition;
    as log10 of the 1e-4 floor it is a jump of ~4 log units next to x = 0.02 that a zero-mean GP cannot follow
    (it overshoots elsewhere). The rows are left out of the pool, the test set and the Pred files.
    Recorded in every metrics row as 'x0_rows'; rows written before the column existed are 'kept'.'''
    return 'dropped' if SYSTEMS[system].get('drop_absent', False) else 'kept'

def row_x0(row):
    '''x0_rows of a metrics row ('kept' for rows written before the column existed).'''
    return row.get('x0_rows') if isinstance(row.get('x0_rows'), str) else 'kept'

def clean_uptake(y):
    y = np.asarray(y, dtype = float)
    return np.where(y <= 0, ZERO_UPTAKE, y)

# 'all'     : MAPE over every test point, zero uptakes set to ZERO_UPTAKE (as in the original scripts)
# 'nonzero' : MAPE over the test points with a GCMC uptake above zero (IRMOF / NU systems). A zero uptake
#             (e.g. CO2 at x = 0) has no relative error; dividing by ZERO_UPTAKE turned one IRMOF-1 miss
#             of 0.19 at a true 0 into an error of 1880
MAPE_RULES = ['all', 'nonzero']

def relative_error(system, y_gcmc, y_hat):
    rule = SYSTEMS[system].get('mape', 'all')
    if rule not in MAPE_RULES:
        raise ValueError(f'Unknown MAPE rule {rule}')
    y_gcmc, y_hat = np.asarray(y_gcmc, dtype = float), np.asarray(y_hat, dtype = float)
    keep = y_gcmc > 0 if rule == 'nonzero' else np.ones(len(y_gcmc), dtype = bool)
    y_ref = clean_uptake(y_gcmc[keep])
    return {'MAPE': np.mean(np.abs((y_hat[keep] - y_ref) / y_ref)), 'n_test_MAPE': int(keep.sum())}    # MAPE as a fraction

# ===================== held-out split =====================

# 'random'   : stratified random 20 % of all grid rows
# 'lowest_p' : rows at the lowest grid pressure always stay in the pool (that pressure is isolated
#              from the next one in log scale, so testing on it is extrapolation); the same number of
#              test rows (20 % of the full grid) is drawn, stratified, from the remaining rows
SPLIT_RULES = ['random', 'lowest_p']

def split_rule(system, rule = None):
    '''Split rule to use: the one asked for, else the system default from systems.py.'''
    rule = rule or SYSTEMS[system].get('split', 'random')
    if rule not in SPLIT_RULES:
        raise ValueError(f'Unknown split rule {rule}')
    return rule

def split_dir(system, space, rule = None):
    '''splits/ holds the system's default rule; any other rule lives in splits_<rule>/.'''
    rule = split_rule(system, rule)
    name = 'splits' if rule == split_rule(system) else f'splits_{rule}'
    return os.path.join(system_dir(system), name, space)

def make_split(system, space, seed, rule = None):
    '''Stratified random hold-out: by temperature for P-X-T grids, by mole fraction for P-X grids.'''
    df, spec = read_grid_file(system, space)
    if spec['T'] is not None:
        strat = df[spec['T']].round(3).values
    else:
        first_gas = list(spec['x'].keys())[0]
        strat = df[spec['x'][first_gas]].round(4).values
    rows = np.arange(len(df))
    if split_rule(system, rule) == 'random':
        __, test_rows = train_test_split(rows, test_size = TEST_FRAC, random_state = seed, stratify = strat)
    else:
        n_test = int(np.ceil(TEST_FRAC * len(df)))     # same test size as the 'random' rule
        P = df[spec['P']].values
        rows = rows[P > P.min()]
        __, test_rows = train_test_split(rows, test_size = n_test, random_state = seed, stratify = strat[rows])
    return np.sort(test_rows)

def load_test_rows(system, space, seed, rule = None):
    path = os.path.join(split_dir(system, space, rule), f'test_idx_seed{seed}.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found - run job_script/make_splits.py first')
    return pd.read_csv(path)['grid_row'].values

# ===================== initial training data =====================

def lhs_rows(pressure, n_samples, seed = LHS_SEED):
    '''Same selection as GPR.train_data_generator: LHS in pressure + nearest neighbour,
    increasing the number of LHS samples until no duplicates remain.
    Returns positions into the given pressure array.'''
    X = np.asarray(pressure, dtype = float).reshape(-1,1)
    no_samples = n_samples
    while True:
        idx = GPR.lhs_sampler(X, no_samples, seed)
        if len(idx) == no_samples:
            return idx
        no_samples += 1

def match_rows(grid, points, tol_x = 1e-3):
    '''Grid row matching the input coordinates of each point, -1 if the point is not on the grid.'''
    rows = []
    for __, pt in points.iterrows():
        mask = np.isclose(grid['P'].values, pt['P'], rtol = 1e-4, atol = 1e-6)
        if 'T' in grid.columns:
            mask &= np.isclose(grid['T'].values, pt['T'], atol = 0.5)
        if 'x' in grid.columns and 'x' in pt.index:
            mask &= np.isclose(grid['x'].values, pt['x'], atol = tol_x)
        found = grid.index[mask]
        if len(found) > 1:
            raise ValueError(f'Point {pt.to_dict()} matched {len(found)} grid rows')
        rows.append(found[0] if len(found) == 1 else -1)
    return np.array(rows, dtype = int)

def load_bi(system, space, gas, grid):
    '''
    BI training set (BI/training_<space>_BI.csv, same header as the grid file).
    The BI points are used with their own inputs and uptakes. Any BI point that is also
    on the GCMC grid is returned in on_grid_rows so it can be removed from pool and test.
    '''
    spec = SYSTEMS[system]['grids'][space]
    path = os.path.join(system_dir(system), 'BI', f'training_{space}_BI.csv')
    bi = pd.read_csv(path)
    pts = pd.DataFrame({'P': bi[spec['P']].values})
    if spec['T'] is not None:
        pts['T'] = bi[spec['T']].values
    pts['x'] = bi[spec['x'][gas]].values
    pts['y'] = bi[spec['y'][gas]].values
    rows = match_rows(grid, pts)
    pts.index = rows
    on_grid_rows = np.unique(rows[rows >= 0])
    print(f'BI set {path}: {len(pts)} points, {len(on_grid_rows)} on the GCMC grid')
    return pts, on_grid_rows

# ===================== scaling =====================

class InputScaler:
    '''Input scaling with statistics taken from the pool only (never from the test set).'''

    def __init__(self, pool):
        self.has_T = 'T' in pool.columns
        self.p_g = pool['P'].values / 1e5
        self.t_g = pool['T'].values if self.has_T else None
        self.x_g = pool['x'].values

    def transform(self, rows):
        cols = [GPR.normalize(rows['P'].values / 1e5, self.p_g, 'LogStandardization')]
        if self.has_T:
            cols.append(GPR.normalize(rows['T'].values, self.t_g, 'LogStandardization'))
        cols.append(GPR.normalize(rows['x'].values, self.x_g, 'MinMaxNorm'))
        return np.column_stack(cols)

def log_uptake(y):
    return np.log10(clean_uptake(y)).reshape(-1,1)

# ===================== active learning =====================

def active_learning(x_train, y_train, x_pool, y_pool, stop_rule, lim, max_iter, mean_func = None):
    '''
    AL loop: fit GP, compute sigma/|mu| on the remaining pool, stop or add the most uncertain point.
    stop_rule = 'max': stop when max(sigma/|mu|) < lim     (GP RE criterion)
    stop_rule = 'pac': stop when PAC >= lim, PAC = % of pool points with sigma/|mu| < PAC_TOL
    Returns the final model, training data, positions (into x_pool) of queried points and the history.
    '''
    remaining = np.arange(len(x_pool))
    picked = []
    history = {'GP MRE': [], 'GP max RE': [], 'PAC': []}
    model = None

    for it in range(max_iter):
        model = GPR.gp(x_train, y_train, config = gpConfig, mean_func = mean_func)
        if len(remaining) == 0:
            warnings.warn('Pool exhausted before the stopping criterion was met')
            break

        y_pred, sigma = GPR.gp_predict(model, x_pool[remaining])
        rel_error = np.abs(np.asarray(sigma)).ravel() / np.abs(np.asarray(y_pred)).ravel()

        PAC = 100 * np.mean(rel_error < PAC_TOL)
        history['GP MRE'].append(np.mean(rel_error))
        history['GP max RE'].append(np.max(rel_error))
        history['PAC'].append(PAC)

        if stop_rule == 'max':
            converged = np.max(rel_error) < lim
        elif stop_rule == 'pac':
            converged = PAC >= lim
        else:
            raise ValueError(f'Unknown stop rule {stop_rule}')
        if converged:
            print(f'Done after {it + 1} GP fits, final max error {np.max(rel_error)}, PAC {PAC}')
            break

        k = np.argmax(rel_error)
        pos = remaining[k]
        picked.append(pos)
        x_train = np.vstack([x_train, x_pool[pos]])
        y_train = np.vstack([y_train, y_pool[pos]])
        remaining = np.delete(remaining, k)
    else:
        warnings.warn(f'AL reached max_iter = {max_iter} without meeting the stopping criterion')

    return model, x_train, y_train, np.array(picked, dtype = int), history

# ===================== TAL source model =====================

def pure_source_model(system, gas, dim, stop_rule, lim, init = 'lhs', n_init = 15):
    '''AL on the pure-component isotherm (source data, not part of the held-out evaluation).'''
    df = pd.read_csv(os.path.join(system_dir(system), SYSTEMS[system]['pure'][gas]))
    P = df[PURE_P].values
    x_g = P / 1e5

    X_all = GPR.normalize(P / 1e5, x_g, 'LogStandardization').reshape(-1,1)
    X_all = GPR.data_padding(X_all, dim)
    Y_all = log_uptake(df[PURE_Y].values)

    if init == 'lhs':
        init_rows = lhs_rows(P, n_init)
        x_init, y_init = X_all[init_rows], Y_all[init_rows]
    else:
        bi = pd.read_csv(os.path.join(system_dir(system), 'BI', f'training_pure_{gas}_BI.csv'))
        rows = match_rows(pd.DataFrame({'P': P}), pd.DataFrame({'P': bi[PURE_P].values}))
        init_rows = np.unique(rows[rows >= 0])
        x_init = GPR.data_padding(GPR.normalize(bi[PURE_P].values / 1e5, x_g, 'LogStandardization').reshape(-1,1), dim)
        y_init = log_uptake(bi[PURE_Y].values)
    pool_rows = np.setdiff1d(np.arange(len(P)), init_rows)

    model, x_tr, __, __, __ = active_learning(x_init, y_init, X_all[pool_rows], Y_all[pool_rows],
                                             stop_rule, lim, MAX_ITER_PURE)
    print(f'Pure {gas} source model trained on {len(x_tr)} points')
    return model, len(x_tr)

def source_mean(source_model):
    '''Original TAL mapping (prior 'original'): the scaled mixture inputs go into the pure GP unchanged.
    Pressure is then read on the wrong scale (mixture-pool instead of pure-isotherm statistics) and the mole
    fraction acts as a distance in the zero-padded columns. Used before 2026-11-07; kept only to reproduce
    the original results (v1.0 of this repository).'''

    class SourceMean(gpflow.mean_functions.MeanFunction):

        def __call__(self, X):
            X = tf.convert_to_tensor(X)
            y_mean, __ = GPR.gp_predict(source_model, X)
            return y_mean

    return SourceMean()

PRIORS = ['xP', 'original']
DEFAULT_PRIOR = 'xP'

def pure_pressure_stats(system, gas):
    '''log10 P [bar] of the pure isotherm: the statistics the pure GP inputs were standardized with, and the range.'''
    df = pd.read_csv(os.path.join(system_dir(system), SYSTEMS[system]['pure'][gas]))
    lp = np.log10(df[PURE_P].values / 1e5)
    return {'mean': np.mean(lp), 'std': np.std(lp), 'lo': np.min(lp), 'hi': np.max(lp)}

def prior_mean(source, scaler, pure_stats, prior, dim):
    '''
    Mean function of the mixture GP built from the pure-component GP (how the pure model enters the mixture).
    The scaled mixture inputs are first turned back into P [bar] and mole fraction x (scaler = the mixture
    InputScaler), then the pure GP is read at an effective pressure standardized with the pure-isotherm
    statistics and zero padded, exactly as the pure GP was trained:
        'xP' : pure isotherm at the partial pressure x P                 log q = f(x P)      (default)
    Outside the pure pressure range: below it Henry's law (log q falls with slope 1 in log P), above it the
    value at the highest pure pressure is held. The prior is floored at log10(ZERO_UPTAKE), the value zero
    uptakes are given. Temperature is not used.
    'original': the mapping used before 2026-11-07 (see source_mean).
    '''
    if prior == 'original':
        return source_mean(source)
    if prior not in PRIORS:
        raise ValueError(f'Unknown prior {prior}')

    lp_pool = np.log10(scaler.p_g)
    p_mean, p_std = np.mean(lp_pool), np.std(lp_pool)
    x_min, x_max = np.min(scaler.x_g), np.max(scaler.x_g)
    x_col = 2 if scaler.has_T else 1
    floor = np.log10(ZERO_UPTAKE)
    ps = pure_stats

    class PriorMean(gpflow.mean_functions.MeanFunction):

        def __call__(self, X):
            X = tf.cast(tf.convert_to_tensor(X), tf.float64)
            logp = X[:, 0] * p_std + p_mean                                          # log10 P [bar]
            x = (X[:, x_col] + 1) / 2 * (x_max - x_min) + x_min                       # undo MinMaxNorm
            logx = tf.math.log(tf.maximum(x, 1e-12)) / np.log(10.0)
            loge = logp + logx                                                        # partial pressure x P
            z = (tf.clip_by_value(loge, ps['lo'], ps['hi']) - ps['mean']) / ps['std']
            Z = tf.concat([z[:, None], tf.zeros([tf.shape(z)[0], dim - 1], dtype = tf.float64)], axis = 1)
            mu, __ = source.predict_f(Z)
            mu = mu[:, 0] + tf.minimum(loge - ps['lo'], 0.0)                          # Henry's law below the range
            return tf.maximum(mu, floor)[:, None]

    return PriorMean()

# ===================== one mixture model =====================

def train_and_evaluate(system, space, gas, seed, tag, init, n_init, stop_rule, lim,
                       exclude_bi_from_test, source = None, prior = DEFAULT_PRIOR, dim = None, extra = None, split = None):
    '''
    Build test/pool for this seed, select the initial points, run AL, then write
        Pred_<tag>_seed<N>.csv     predictions on the full GCMC grid with a 'split' column
        Sampled_<tag>_seed<N>.csv  final training points
        History_<tag>_seed<N>.csv  GP MRE / max RE / PAC per GP fit
    and return a dictionary of test-set metrics.
    source: pure-component GP for TAL (its mean enters the mixture GP as set by prior, see prior_mean);
            None = plain AL (zero mean).
    split: split rule ('random' / 'lowest_p'), None = system default from systems.py.
    '''
    grid = load_grid(system, space, gas)
    all_rows = grid.index.values
    split = split_rule(system, split)
    full_test_rows = np.intersect1d(load_test_rows(system, space, seed, split), all_rows)    # rows left out of the grid drop out
    pool_rows = np.setdiff1d(all_rows, full_test_rows)

    # BI points on the grid are removed from the test set (for both M and M1 runs)
    if init == 'bi' or exclude_bi_from_test:
        bi_pts, bi_rows = load_bi(system, space, gas, grid)
        test_rows = np.setdiff1d(full_test_rows, bi_rows)
    else:
        test_rows = full_test_rows

    if init == 'lhs':
        init_rows = pool_rows[lhs_rows(grid.loc[pool_rows, 'P'].values, n_init)]
        init_pts = grid.loc[init_rows]
    elif init == 'bi':
        init_rows = bi_rows
        init_pts = bi_pts
    else:
        raise ValueError(f'Unknown init {init}')
    if np.intersect1d(init_rows, test_rows).size > 0:
        raise RuntimeError('Initial training points overlap the test set')

    # AL can only query pool points that are not already in the training set
    query_rows = np.setdiff1d(pool_rows, init_rows)

    scaler = InputScaler(grid.loc[pool_rows])
    dim = dim or (3 if 'T' in grid.columns else 2)

    def encode(points):
        return GPR.data_padding(scaler.transform(points), dim)

    x_init, y_init = encode(init_pts), log_uptake(init_pts['y'])
    x_query, y_query = encode(grid.loc[query_rows]), log_uptake(grid.loc[query_rows, 'y'])

    mean_func = None if source is None else prior_mean(source, scaler, pure_pressure_stats(system, gas), prior, dim)
    model, x_train, __, picked, history = active_learning(x_init, y_init, x_query, y_query,
                                                          stop_rule, lim, MAX_ITER_MIX, mean_func)
    al_rows = query_rows[picked]

    # predictions on the full GCMC grid
    mu, sigma = GPR.gp_predict(model, encode(grid))
    pred = 10 ** np.asarray(mu).ravel()

    labels = pd.Series('pool_unsampled', index = all_rows)
    labels[test_rows] = 'test'
    labels[np.setdiff1d(full_test_rows, test_rows)] = 'test_excluded_bi'
    labels[init_rows] = 'train_initial'
    labels[al_rows] = 'train_AL'

    out = pd.DataFrame({'Pressure(Pa)': grid['P'].values})
    if 'T' in grid.columns:
        out['Temperature(K)'] = grid['T'].values
    out['mole fraction'] = grid['x'].values
    out['GCMC'] = grid['y'].values
    out['Predicted'] = pred
    out['sigma_log10'] = np.asarray(sigma).ravel()
    out['split'] = labels.values
    out.to_csv(f'Pred_{tag}_seed{seed}.csv', index = False)

    # grid_row = -1 marks BI points that are not on the GCMC grid
    sampled = pd.concat([init_pts.assign(source = 'initial'), grid.loc[al_rows].assign(source = 'AL')])
    sampled.to_csv(f'Sampled_{tag}_seed{seed}.csv', index_label = 'grid_row')
    pd.DataFrame(history).to_csv(f'History_{tag}_seed{seed}.csv', index = False)

    # metrics on the held-out test set only
    y_true = clean_uptake(grid.loc[test_rows, 'y'].values)
    y_hat = pd.Series(pred, index = all_rows).loc[test_rows].values    # pred follows the grid rows (index may have gaps)
    metrics = {'seed': seed, 'space': space, 'split_rule': split, 'x0_rows': x0_rule(system), 'init': init,
               'n_init': len(init_pts), 'n_train_final': len(x_train), 'n_iters': len(history['GP MRE']),
               'n_test': len(test_rows), 'n_bi_removed_from_test': len(full_test_rows) - len(test_rows),
               'R2': r2_score(y_true, y_hat),
               'MAE': mean_absolute_error(y_true, y_hat),
               **relative_error(system, grid.loc[test_rows, 'y'].values, y_hat),
               'final_GP_max_RE': history['GP max RE'][-1] if history['GP max RE'] else np.nan,
               'final_PAC': history['PAC'][-1] if history['PAC'] else np.nan}
    if extra:
        metrics.update(extra)
    print(metrics)
    return metrics

def write_metrics(rows, name, seed):
    pd.DataFrame(rows).to_csv(f'Metrics_{name}_seed{seed}.csv', index = False)

def read_cfg(script):
    '''cfg dict of a run script, read without running it.'''
    tree = ast.parse(open(script).read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', None) == 'cfg':
            return ast.literal_eval(node.value)
    raise ValueError(f'no cfg in {script}')

# ===================== run types =====================

def run_al(name, cfg, seed):
    '''Plain AL for one gas, for every space in cfg['spaces'].'''
    rows = []
    for space in cfg['spaces']:
        tag = name if len(cfg['spaces']) == 1 else f'{name}_{space}'
        rows.append(train_and_evaluate(cfg['system'], space, cfg['gas'], seed, tag,
                                       cfg['init'], cfg['n_init'].get(space), cfg['stop_rule'], cfg['lim'],
                                       cfg.get('exclude_bi_from_test', False), split = cfg.get('split')))
    write_metrics(rows, name, seed)

def tal_dim(system, spaces):
    '''Input dimension shared by the pure model and every mixture space of one TAL run, as in the
    original scripts: 3 when any space has temperature (px inputs then get a zero third column), else 2.'''
    return 3 if any(SYSTEMS[system]['grids'][s]['T'] is not None for s in spaces) else 2

def run_tal(name, cfg, seed):
    '''TAL: pure-component source GP as the mean function of the mixture GP (cfg['prior'], see prior_mean),
    for every space in cfg['spaces']. One pure model is trained per run and shared by all spaces.'''
    rows = []
    dim = tal_dim(cfg['system'], cfg['spaces'])
    prior = cfg.get('prior', DEFAULT_PRIOR)
    source, n_pure = pure_source_model(cfg['system'], cfg['gas'], dim, cfg['pure_stop_rule'], cfg['pure_lim'],
                                       init = cfg['init'], n_init = cfg.get('n_pure', 15))
    for space in cfg['spaces']:
        tag = f'{name}_{space}'
        rows.append(train_and_evaluate(cfg['system'], space, cfg['gas'], seed, tag,
                                       cfg['init'], cfg['n_init'].get(space), cfg['stop_rule'], cfg['lim'],
                                       cfg.get('exclude_bi_from_test', False),
                                       source = source, prior = prior, dim = dim,
                                       extra = {'prior': prior, 'n_pure_train': n_pure}, split = cfg.get('split')))
    write_metrics(rows, name, seed)

def row_prior(row):
    '''Prior a metrics row was made with; rows written before the column existed used the original mapping.'''
    return row.get('prior') if isinstance(row.get('prior'), str) else 'original'

def run_data_size(name, cfg, seed):
    '''TAL repeated for every initial training size in cfg['sizes'][space] (LHS from the pool).
    Sizes already in Metrics_<name>_<space>_seed<N>.csv (from an earlier, killed job) are skipped.
    One pure model is trained per run (only if something is left to do) and shared by all spaces and sizes.'''
    dim = tal_dim(cfg['system'], cfg['spaces'])
    split = split_rule(cfg['system'], cfg.get('split'))
    prior = cfg.get('prior', DEFAULT_PRIOR)
    source = None
    for space in cfg['spaces']:
        done_file = f'Metrics_{name}_{space}_seed{seed}.csv'
        rows = pd.read_csv(done_file).to_dict('records') if os.path.exists(done_file) else []
        # only results made with the same split rule (files without the column are 'random'), prior and x = 0 rule
        # count as done
        rows = [r for r in rows if (r.get('split_rule') if isinstance(r.get('split_rule'), str) else 'random') == split
                and row_prior(r) == prior and row_x0(r) == x0_rule(cfg['system'])]
        done = {int(r['n_requested']) for r in rows}
        todo = [size for size in cfg['sizes'][space] if size not in done]
        print(f'{space} seed {seed}: sizes done {sorted(done)}, to run {todo}', flush = True)
        if not todo:
            continue
        if source is None:
            source, n_pure = pure_source_model(cfg['system'], cfg['gas'], dim, cfg['pure_stop_rule'], cfg['pure_lim'],
                                               init = 'lhs', n_init = cfg.get('n_pure', 15))
        for size in todo:
            rows.append(train_and_evaluate(cfg['system'], space, cfg['gas'], seed, f'{name}_{space}_n{size}',
                                           'lhs', size, cfg['stop_rule'], cfg['lim'], False,
                                           source = source, prior = prior, dim = dim,
                                           extra = {'n_requested': size, 'prior': prior, 'n_pure_train': n_pure},
                                           split = split))
            # write after every size so partial results survive a killed job
            write_metrics(rows, f'{name}_{space}', seed)

# ===================== ablation: transfer vs ordinary GP at fixed sizes =====================

# Arms: 'GP' = ordinary GP (zero mean, as in AL); 'TGP_<prior>' = GP with the pure-component mean entering
# as <prior> (see prior_mean): 'TGP_xP', as in TAL.
DEFAULT_ARMS = ['GP', f'TGP_{DEFAULT_PRIOR}']

def arm_prior(arm):
    '''Prior of an ablation arm, None for the ordinary GP.'''
    if arm == 'GP':
        return None
    if not arm.startswith('TGP_') or arm[4:] not in PRIORS:
        raise ValueError(f'Unknown ablation arm {arm}')
    return arm[4:]

def ablation_cases(sizes, arms = DEFAULT_ARMS):
    '''(arm, size) pairs of one space. Size 0 is the pure-component mean alone, so it exists for TGP arms only.'''
    return [(arm, n) for n in sizes for arm in arms if n > 0 or arm_prior(arm) is not None]

def ablation_data(system, space, gas, seed, split, exclude_bi_from_test, dim):
    '''Test set and a random ordering of the pool for one split seed, encoded as in train_and_evaluate.
    The ordering (draw seed = split seed) is shared by every size and both arms: the training set of size n
    is the first n pool points, so the sets are nested and GP / TGP are trained on exactly the same points.'''
    grid = load_grid(system, space, gas)
    full_test_rows = np.intersect1d(load_test_rows(system, space, seed, split), grid.index.values)    # rows left out drop out
    pool_rows = np.setdiff1d(grid.index.values, full_test_rows)
    if exclude_bi_from_test:
        __, bi_rows = load_bi(system, space, gas, grid)
        test_rows = np.setdiff1d(full_test_rows, bi_rows)
    else:
        test_rows = full_test_rows
    order = np.random.default_rng(seed).permutation(pool_rows)

    scaler = InputScaler(grid.loc[pool_rows])

    def encode(points):
        return GPR.data_padding(scaler.transform(points), dim)

    return {'x_pool': encode(grid.loc[order]), 'y_pool': log_uptake(grid.loc[order, 'y']),
            'x_test': encode(grid.loc[test_rows]), 'y_test': grid.loc[test_rows, 'y'].values,
            'n_test': len(test_rows), 'n_bi_removed_from_test': len(full_test_rows) - len(test_rows),
            'scaler': scaler}

def ablation_fit(system, gas, data, arm, n, source, dim):
    '''One GP fit (no AL) on the first n points of the pool ordering; test-set R2 / MAE / MAPE.
    n = 0: the prediction is the pure-component mean itself.'''
    if n > len(data['x_pool']):
        raise ValueError(f'Size {n} is larger than the pool ({len(data["x_pool"])} points)')
    prior = arm_prior(arm)
    mean_func = None if prior is None else prior_mean(source, data['scaler'], pure_pressure_stats(system, gas), prior, dim)
    if n == 0:
        mu = mean_func(data['x_test'])
    else:
        model = GPR.gp(data['x_pool'][:n], data['y_pool'][:n], config = gpConfig, mean_func = mean_func)
        mu, __ = GPR.gp_predict(model, data['x_test'])
    y_hat = 10 ** np.asarray(mu).ravel()
    y_true = clean_uptake(data['y_test'])
    return {'R2': r2_score(y_true, y_hat), 'MAE': mean_absolute_error(y_true, y_hat),
            **relative_error(system, data['y_test'], y_hat)}

def run_ablation(name, cfg, seed):
    '''
    Ablation of the transfer, without AL: for every size in cfg['sizes'][space] and every arm in cfg['arms']
    (ordinary GP 'GP', GP with the pure-component mean 'TGP_<prior>', same pure model as the TAL runs), one GP
    fit on the same random, nested training set from the pool (see ablation_data). Size 0 is the pure-component
    mean alone (TGP arms only). Writes Metrics_<name>_<space>_seed<N>.csv, one row per arm and size, after every
    fit; cases already in that file (same split rule and x = 0 rule) are skipped, so a killed job can simply be
    resubmitted. Metrics only, no Pred files.
    '''
    dim = tal_dim(cfg['system'], cfg['spaces'])
    split = split_rule(cfg['system'], cfg.get('split'))
    x0 = x0_rule(cfg['system'])
    arms = cfg.get('arms', DEFAULT_ARMS)
    source, n_pure = None, None
    for space in cfg['spaces']:
        done_file = f'Metrics_{name}_{space}_seed{seed}.csv'
        rows = pd.read_csv(done_file).to_dict('records') if os.path.exists(done_file) else []
        rows = [r for r in rows if r.get('split_rule') == split and row_x0(r) == x0]
        done = {(r['arm'], int(r['n_requested'])) for r in rows}
        todo = [case for case in ablation_cases(cfg['sizes'][space], arms) if case not in done]
        print(f'{space} seed {seed}: {len(done)} cases done, to run {todo}', flush = True)
        if not todo:
            continue
        if source is None:
            source, n_pure = pure_source_model(cfg['system'], cfg['gas'], dim, cfg['pure_stop_rule'], cfg['pure_lim'],
                                               init = 'lhs', n_init = cfg.get('n_pure', 15))
        data = ablation_data(cfg['system'], space, cfg['gas'], seed, split, cfg.get('exclude_bi_from_test', False), dim)
        for arm, n in todo:
            prior = arm_prior(arm)
            metrics = {'seed': seed, 'space': space, 'split_rule': split, 'x0_rows': x0, 'arm': arm, 'n_requested': n,
                       'init': 'random', 'n_test': data['n_test'], 'n_bi_removed_from_test': data['n_bi_removed_from_test'],
                       **ablation_fit(cfg['system'], cfg['gas'], data, arm, n, source, dim),
                       'prior': prior or 'none', 'n_pure_train': n_pure if prior else np.nan}
            print(metrics, flush = True)
            rows.append(metrics)
            write_metrics(rows, f'{name}_{space}', seed)
