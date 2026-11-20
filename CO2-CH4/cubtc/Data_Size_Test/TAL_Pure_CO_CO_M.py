'''
Data-size test: TAL for CO2 (px, pxt) for several LHS initial sizes, held-out test set
Usage: python TAL_Pure_CO_CO_M.py <seed>      (seed = held-out split 0-4; job_script/run_all.sh runs all five)
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
    'system': 'CO2-CH4/cubtc',
    'gas': 'CO2',
    'spaces': ['px', 'pxt'],
    'sizes': {'px': [5, 10, 15, 20, 25, 30, 35, 40], 'pxt': [5, 10, 20, 25, 40]},    # initial LHS training sizes to test
    'n_pure': 15,    # initial pure-isotherm training size
    'stop_rule': 'max',    # 'max': stop when max GP RE < lim, 'pac': stop when PAC >= lim
    'lim': 0.02,
    'pure_stop_rule': 'max',    # stopping rule of the pure-component source model
    'pure_lim': 0.02,
    'prior': 'xP',    # how the pure GP enters the mixture GP: pure isotherm at the partial pressure x P (heldout.prior_mean)
}

heldout.run_data_size('TAL_Pure_CO_CO_M', cfg, seed)
