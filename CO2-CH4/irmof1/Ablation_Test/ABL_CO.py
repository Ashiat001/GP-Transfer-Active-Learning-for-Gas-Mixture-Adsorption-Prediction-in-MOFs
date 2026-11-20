'''
Ablation test: ordinary GP vs GP with the pure-component mean (transfer), CO2 (PX), held-out test set.
Every arm is fitted once (no AL) on the same random, nested training sets from the pool; size 0 = pure-component mean only.
Usage: python ABL_CO.py <seed>      (seed = held-out split 0-4; job_script/run_all.sh runs all five)
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
    'system': 'CO2-CH4/irmof1',
    'gas': 'CO2',
    'spaces': ['px'],
    'sizes': {'px': [0, 5, 10, 20, 40, 80, 160, 320, 640]},    # mixture training sizes (random from the pool, nested); 0 = pure-component mean only
    'n_pure': 15,    # initial pure-isotherm training size (LHS), as in the TAL M runs
    'pure_stop_rule': 'max',    # stopping rule of the pure-component source model (GP RE)
    'pure_lim': 0.02,
    'arms': ['GP', 'TGP_xP'],    # ordinary GP; pure mean at the partial pressure x P (as in TAL)
    'exclude_bi_from_test': True,    # remove BI points from the test set, so it matches the training-data / AF tests
}

heldout.run_ablation('ABL_CO', cfg, seed)
