'''
AL for Xe (pxt) with held-out test set, initial data: LHS
Usage: python AL_Xe_pxt.py <seed>      (seed = held-out split 0-4; job_script/run_all.sh runs all five)
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
    'system': 'Xe-Kr',
    'gas': 'Xe',
    'spaces': ['pxt'],
    'init': 'lhs',    # 'lhs' = M (LHS from the pool), 'bi' = M1 (Datasets/.../BI/training_<space>_BI.csv)
    'n_init': {'pxt': 20},    # initial training size (used for LHS only)
    'stop_rule': 'pac',    # 'max': stop when max GP RE < lim, 'pac': stop when PAC >= lim
    'lim': 90,
    'exclude_bi_from_test': False,    # remove BI points from the test set (M and M1 share one test set)
}

heldout.run_al('AL_Xe_pxt', cfg, seed)
