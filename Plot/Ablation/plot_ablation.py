'''Ablation plots: test error vs mixture training data size, ordinary GP vs GP with the pure-component mean (TGP, xP).
Point = mean over the 5 data splits, error bar = standard error of the mean (std / sqrt(5)), same as the data-size plots.
Size 0 = pure-component mean alone (TGP only; GP has nothing to predict with). Sizes double, so the x axis is evenly spaced.
One figure per component (panels = P-X, P-X-T): <OUT>/<system>/Ablation_test_<gas>.png and .pdf.
Reads the per-seed rows in <system>/Ablation_Test/All_metrics.csv (read only).
Usage: python Plot/Ablation/plot_ablation.py'''
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     # repository folder
OUT = os.path.dirname(os.path.abspath(__file__))     # figures go to <folder of this script>/<system>/
METRIC = 'MAE'                                       # 'MAE' (mg/g) or 'MAPE' (shown in %)
DPI = 600

# gas: (run folder, file name, title label)
CO2 = ('ABL_CO', 'CO2', r'$\mathbf{CO_2}$')
CH4 = ('ABL_CH', 'CH4', r'$\mathbf{CH_4}$')
H2S = ('ABL_HS', 'H2S', r'$\mathbf{H_2S}$')
KR = ('ABL_Kr', 'Kr', r'$\mathbf{Kr}$')
XE = ('ABL_Xe', 'Xe', r'$\mathbf{Xe}$')
# system folder: ([gases], spaces). Remove a gas from a list to skip its figure.
SYSTEMS = {
    'CO2-CH4/cubtc':   ([CO2, CH4], ['px', 'pxt']),
    'CO2-CH4/irmof1':  ([CO2, CH4], ['px']),
    'CO2-CH4/irmof10': ([CO2, CH4], ['px']),
    'CO2-CH4/nu800':   ([CO2, CH4], ['px']),
    'CO2-H2S':         ([CO2, H2S], ['px', 'pxt']),
    'Xe-Kr':           ([KR, XE], ['px', 'pxt']),
}
SPACE = {'px': 'P-X', 'pxt': 'P-X-T'}
MOF = {'CO2-CH4/cubtc': 'Cu-BTC', 'CO2-CH4/irmof1': 'IRMOF-1', 'CO2-CH4/irmof10': 'IRMOF-10',
       'CO2-CH4/nu800': 'NU-800', 'CO2-H2S': 'Cu-BTC', 'Xe-Kr': 'Cu-BTC'}    # panel title: MOF, space
# two-panel figures get (a), (b); a single-panel figure gets a letter only if listed here (system, gas file name)
SINGLE_LABEL = {('CO2-CH4/irmof1', 'CO2'): '(c)'}
ARMS = {    # arm: (legend label, colour, edge colour, marker, small x shift so the bars don't overlap)
    'GP':     ('GP (no transfer)', '#B2182B', '#67001F', 's', -0.08),
    'TGP_xP': ('TGP', '#2166AC', '#053061', 'o', 0.08),
}
YLABEL = {'MAE': 'Test MAE (mg/g)', 'MAPE': 'Test MAPE (%)'}[METRIC]


def stats(system, run, space):
    '''mean and standard error over the 5 splits, per arm and size'''
    m = pd.read_csv(os.path.join(ROOT, system, 'Ablation_Test', 'All_metrics.csv'))
    m = m[(m['run'] == run) & (m['file'] == f'Metrics_{run}_{space}')]     # rows of Metrics_<run>_<space>_seed<N>.csv
    assert m['seed'].nunique() == 5, (system, run, space)
    m = m[m['arm'].isin(list(ARMS))]
    m['val'] = m['MAPE'] * 100 if METRIC == 'MAPE' else m['MAE']
    g = m.groupby(['arm', 'n_requested'])['val']
    s = pd.DataFrame({'mean': g.mean(), 'se': g.std() / np.sqrt(g.count()), 'median': g.median(),
                      'max': g.max(), 'n_seeds': g.count()}).reset_index()
    for _, r in s[s['max'] > 10 * s['median']].iterrows():    # one split far off: worth a line in the caption
        print(f"  note {system} {run} {space} {r.arm} n={r.n_requested}: one split at {r['max']:.4g}, "
              f"mean {r['mean']:.4g} vs median {r['median']:.4g}")
    return s


for system, (gases, spaces) in SYSTEMS.items():
    for run, name, gas in gases:                       # one figure per component
        # P-X and P-X-T side by side (15 x 6, like the data-size plots); P-X only: one panel (8 x 6)
        fig, ax = plt.subplots(1, len(spaces), figsize = (15, 6) if len(spaces) > 1 else (8, 6), squeeze = False)
        ax = ax.ravel()

        for k, sp in enumerate(spaces):
            a = ax[k]
            st = stats(system, run, sp)
            sizes = sorted(st['n_requested'].unique())
            pos = {s: i for i, s in enumerate(sizes)}

            for arm, (label, col, edge, marker, dodge) in ARMS.items():
                d = st[st['arm'] == arm].sort_values('n_requested')
                x = d['n_requested'].map(pos) + dodge
                a.plot(x, d['mean'], color = col, linestyle = '--', linewidth = 2.0, zorder = 4)
                # Standard error bars (std / sqrt(5) over the data splits)
                a.errorbar(x, d['mean'], yerr = d['se'], fmt = 'none', ecolor = edge, elinewidth = 1.5,
                           capsize = 4, capthick = 1.5, zorder = 5)
                a.scatter(x, d['mean'], s = 90, marker = marker, color = col, alpha = 0.9, edgecolors = edge, zorder = 6)

            # log y axis (error falls by orders of magnitude); limits from the bar ends
            lo_end = (st['mean'] - st['se']).where(st['mean'] - st['se'] > 0, st['mean'])
            lo, hi = lo_end.min(), (st['mean'] + st['se']).max()
            a.set_yscale('log')
            a.set_ylim(lo / 1.4, hi * 1.4)
            a.yaxis.set_major_locator(LogLocator(base = 10, subs = (1, 2, 5) if hi / lo < 300 else (1,)))
            a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}' if v < 1e4 else f'{v:,.0f}'))
            a.yaxis.set_minor_formatter(NullFormatter())

            a.set_xticks(range(len(sizes)))
            a.set_xticklabels(sizes)
            a.set_xlim(-0.5, len(sizes) - 0.5)
            a.set_xlabel('Mixture Training Data Size', fontsize = 15, fontweight = 'bold')
            a.set_ylabel(YLABEL, fontsize = 15, fontweight = 'bold')
            for tick in a.get_xticklabels() + a.get_yticklabels():
                tick.set_fontsize(13)
                tick.set_fontweight('bold')
            a.set_title(f'{MOF[system]}, {SPACE[sp]}', fontsize = 15, fontweight = 'bold')
            a.grid(True, linestyle = '--', alpha = 0.4)
            panel = f'({chr(97 + k)})' if len(spaces) > 1 else SINGLE_LABEL.get((system, name))
            if panel:
                a.text(-0.10, 1.03, panel, transform = a.transAxes, fontsize = 14, fontweight = 'bold',
                       va = 'bottom', ha = 'left', clip_on = False)     # above the top tick label

        handles = [Line2D([], [], color = col, linestyle = '--', linewidth = 2.0, marker = marker, markersize = 9,
                          markerfacecolor = col, markeredgecolor = edge, alpha = 0.9, label = label)
                   for arm, (label, col, edge, marker, _) in ARMS.items()]
        fig.legend(handles = handles, loc = 'upper center', ncol = 2, fontsize = 13, frameon = False,
                   bbox_to_anchor = (0.5, 1.0))
        plt.tight_layout(rect = [0, 0, 1, 0.93])

        out = os.path.join(OUT, system)
        os.makedirs(out, exist_ok = True)
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(out, f'Ablation_test_{name}.{ext}'), dpi = DPI, bbox_inches = 'tight',
                        pad_inches = 0.05, facecolor = 'white')
        plt.close(fig)
        print('written', os.path.join(out, f'Ablation_test_{name}.png'))
