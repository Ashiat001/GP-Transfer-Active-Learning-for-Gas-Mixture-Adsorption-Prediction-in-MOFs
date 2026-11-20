'''
Description of every mixture system: where the GCMC grids live, which columns hold
pressure / temperature / mole fraction / uptake for each gas, and the pure-component
isotherms used as the TAL source model.

The system name is the folder of its analyses (e.g. CO2-CH4/cubtc); 'dir' is its data folder
in Datasets/. File names are relative to 'dir'.
'''

# pure isotherm files all share the same header
PURE_P = 'Pressure [Pa]'
PURE_Y = 'Uptake [mg/g]'

# Cu-BTC grids (Mukherjee et al.) share one header style
def _cubtc_grid(file, gases, temperature):
    return {
        'file': file,
        'P': 'Pressure(Pa)',
        'T': 'Temperature(K)' if temperature else None,
        'x': {g: f'{g}_mole_frac' for g in gases},
        'y': {g: f'{g}_uptake' for g in gases},
    }

# grids of IRMOF-1, IRMOF-10 and NU-800
_IRMOF_GRID = {
    'file': 'Mixture_px.csv',
    'P': 'Pressure [Pa]',
    'T': None,
    'x': {'CO2': 'Mole Fraction CO2', 'CH4': 'Mole fraction CH4'},
    'y': {'CO2': 'Uptake_CO2 [mg/gr]', 'CH4': 'Uptake_CH4 [mg/gr]'},
}

SYSTEMS = {
    'CO2-CH4/cubtc': {
        'dir': 'Datasets/Cu-BTC/CO2-CH4',
        'grids': {
            'px': _cubtc_grid('Mixture_px.xlsx', ['CO2', 'CH4'], temperature = False),
            'pxt': _cubtc_grid('Mixture_pxt.csv', ['CO2', 'CH4'], temperature = True),
        },
        'pure': {'CO2': 'Pure_CO2_data.csv', 'CH4': 'Pure_CH4_data.csv'},
    },
    'CO2-CH4/irmof1': {
        'dir': 'Datasets/IRMOF-1',
        'grids': {'px': _IRMOF_GRID},
        'split': 'lowest_p',    # 1 Pa rows always in the pool (see heldout.SPLIT_RULES)
        'mape': 'nonzero',      # MAPE over test points with uptake > 0 (see heldout.MAPE_RULES)
        'drop_absent': True,    # grid rows where the modelled gas is absent (x = 0) are left out (see heldout.x0_rule)
        'pure': {'CO2': 'Pure_CO2_data.csv', 'CH4': 'Pure_CH4_data.csv'},
    },
    'CO2-CH4/irmof10': {
        'dir': 'Datasets/IRMOF-10',
        'grids': {'px': _IRMOF_GRID},
        'split': 'lowest_p',    # 1 Pa rows always in the pool (see heldout.SPLIT_RULES)
        'mape': 'nonzero',      # MAPE over test points with uptake > 0 (see heldout.MAPE_RULES)
        'drop_absent': True,    # grid rows where the modelled gas is absent (x = 0) are left out (see heldout.x0_rule)
        'pure': {'CO2': 'Pure_CO2_data.csv', 'CH4': 'Pure_CH4_data.csv'},
    },
    'CO2-CH4/nu800': {
        'dir': 'Datasets/NU-800',
        'grids': {'px': _IRMOF_GRID},
        'split': 'lowest_p',    # 1 Pa rows always in the pool (see heldout.SPLIT_RULES)
        'mape': 'nonzero',      # MAPE over test points with uptake > 0 (see heldout.MAPE_RULES)
        'drop_absent': True,    # grid rows where the modelled gas is absent (x = 0) are left out (see heldout.x0_rule)
        'pure': {'CO2': 'Pure_CO2_data.csv', 'CH4': 'Pure_CH4_data.csv'},
    },
    'CO2-H2S': {
        'dir': 'Datasets/Cu-BTC/CO2-H2S',
        'grids': {
            'px': _cubtc_grid('Mixture_px.csv', ['CO2', 'H2S'], temperature = False),
            'pxt': _cubtc_grid('Mixture_pxt.csv', ['CO2', 'H2S'], temperature = True),
        },
        'pure': {'CO2': 'Pure_CO2_data.csv', 'H2S': 'Pure_H2S_data.csv'},
    },
    'Xe-Kr': {
        'dir': 'Datasets/Cu-BTC/Xe-Kr',
        'grids': {
            'px': _cubtc_grid('Mixture_px.csv', ['Xe', 'Kr'], temperature = False),
            'pxt': _cubtc_grid('Mixture_pxt.csv', ['Xe', 'Kr'], temperature = True),
        },
        'pure': {'Xe': 'Pure_Xe_data.csv', 'Kr': 'Pure_Kr_data.csv'},
    },
}

# held-out split settings shared by every analysis
# (a system without a 'split' entry uses the 'random' rule, without a 'mape' entry the 'all' rule)
TEST_FRAC = 0.2
SEEDS = [0, 1, 2, 3, 4]
