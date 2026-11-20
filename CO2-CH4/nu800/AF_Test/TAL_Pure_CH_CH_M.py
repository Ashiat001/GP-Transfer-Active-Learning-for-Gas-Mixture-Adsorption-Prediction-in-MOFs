'''
TAL (pure CH4 source -> mixture CH4) for px with held-out test set, initial data: LHS
Usage: python TAL_Pure_CH_CH_M.py <seed>      (seed = held-out split 0-4; job_script/run_all.sh runs all five)
'''
import os
import sys

# locate job_script/
here = os.path.dirname(os.path.abspath(__file__))
while not os.path.isdir(os.path.join(here, 'job_script')):
    if os.path.dirname(here) == here:
        sys.exit('Could not find the job_script folder')
    here = os.path.dirname(here)
sys.path.insert(0, os.path.join(here, 'job_script'))
import heldout

seed = int(sys.argv[1])

# ===== settings =====
cfg = {
    'system': 'CO2-CH4/nu800',
    'gas': 'CH4',
    'spaces': ['px'],
    'init': 'lhs',    # 'lhs' = M (LHS from the pool), 'bi' = M1 (Datasets/.../BI/ files)
    'n_init': {'px': 5},    # initial mixture training size per space (LHS only)
    'n_pure': 15,    # initial pure-isotherm training size (LHS only)
    'stop_rule': 'pac',    # 'max': stop when max GP RE < lim, 'pac': stop when PAC >= lim
    'lim': 90,
    'pure_stop_rule': 'max',    # stopping rule of the pure-component source model (GP RE, as in co2-h2s / xe-kr)
    'pure_lim': 0.02,
    'prior': 'xP',    # how the pure GP enters the mixture GP: pure isotherm at the partial pressure x P (heldout.prior_mean)
    'exclude_bi_from_test': True,    # remove BI points from the test set (M and M1 share one test set)
}

heldout.run_tal('TAL_Pure_CH_CH_M', cfg, seed)
