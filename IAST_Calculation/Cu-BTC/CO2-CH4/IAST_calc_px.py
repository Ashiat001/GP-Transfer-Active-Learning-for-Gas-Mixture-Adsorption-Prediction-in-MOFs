import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyiast
import warnings
warnings.filterwarnings('ignore')
import os

# inputs: the GCMC data in Datasets/ (outputs are written to the folder you run this script from)
DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../Datasets/Cu-BTC/CO2-CH4'))
 
#====== Read in Pure and Mixture Dataset =======
df = pd.read_excel(os.path.join(DATA, 'Mixture_px.xlsx'))
df2 = pd.read_csv(os.path.join(DATA, 'Pure_CH4_data.csv'))
df3 = pd.read_csv(os.path.join(DATA, 'Pure_CO2_data.csv')) 
 
# Convert pressure from Pa to bar 
df['Pressure(bar)'] = df['Pressure(Pa)'] / 1e5
df2['Pressure [bar]'] = df2['Pressure [Pa]'] / 1e5
df3['Pressure [bar]'] = df3['Pressure [Pa]'] / 1e5

#====== Fit Pure Component Isotherms ======
print("Fitting CO2 isotherm...")
isotherm_co2 = pyiast.InterpolatorIsotherm(df3,
    loading_key='Uptake [mg/g]',
    pressure_key='Pressure [bar]',
    fill_value=df3['Uptake [mg/g]'].max()
)
# isotherm_co2 = pyiast.ModelIsotherm(df3,
#     loading_key='Uptake [mg/g]',
#     pressure_key='Pressure [bar]',
#     model='DSLangmuir'
# )
 
print("\nFitting CH4 isotherm...")
isotherm_ch4 = pyiast.InterpolatorIsotherm(
    df2,
    loading_key='Uptake [mg/g]',
    pressure_key='Pressure [bar]',
    fill_value=df2['Uptake [mg/g]'].max()
)

# isotherm_ch4 = pyiast.ModelIsotherm(
#     df2,
#     loading_key='Uptake [mg/g]',
#     pressure_key='Pressure [bar]',
#     model='Langmuir'
# )

# # Pre-fetch the lowest data point for Henry's law fallback
co2_pp_min = df3['Pressure [bar]'].iloc[0]
co2_ld_min = df3['Uptake [mg/g]'].iloc[0]
ch4_pp_min = df2['Pressure [bar]'].iloc[0]
ch4_ld_min = df2['Uptake [mg/g]'].iloc[0]
 
KH_co2 = co2_ld_min / co2_pp_min if co2_pp_min > 0 else 0.0
KH_ch4 = ch4_ld_min / ch4_pp_min if ch4_pp_min > 0 else 0.0

#====== Implement IAST for Full Pressure and Mole Fraction Range ======

iast_co2_list   = []
iast_ch4_list   = []
x_co2_list      = []
failed_count     = 0
henry_count      = 0

def try_iast(pp_co2, pp_ch4, y_co2, y_ch4, isotherm_co2, isotherm_ch4):
    for guess in ([y_co2, y_ch4], [0.5, 0.5], [0.99, 0.01], [0.01, 0.99]):
        try:
            return pyiast.iast(
                [pp_co2, pp_ch4],
                [isotherm_co2, isotherm_ch4],
                adsorbed_mole_fraction_guess=guess,
            )
        except Exception:
            continue
    return None  # all guesses failed
 
for _, row in df.iterrows():
    p_total = row['Pressure(bar)']
    y_co2   = row['CO2_mole_frac']
    y_ch4   = row['CH4_mole_frac']
    
    pp_co2  = y_co2 * p_total
    pp_ch4  = y_ch4 * p_total
 
    # Henry's law fallback for partial pressures below the isotherm data range
    if pp_co2 < co2_pp_min or pp_ch4 < ch4_pp_min:
        q_co2 = KH_co2 * pp_co2
        q_ch4 = KH_ch4 * pp_ch4
        total = q_co2 + q_ch4
        x_co2 = q_co2 / total if total > 0 else np.nan
        iast_co2_list.append(q_co2)
        iast_ch4_list.append(q_ch4)
        x_co2_list.append(x_co2)
        henry_count += 1
        continue
    try:
        q = try_iast(pp_co2, pp_ch4, y_co2, y_ch4, isotherm_co2, isotherm_ch4)
        if q is not None:
            q_co2, q_ch4 = q[0], q[1]
            total_q = q_co2 + q_ch4
            x_co2 = q_co2 / total_q if total_q > 0 else np.nan
            iast_co2_list.append(q_co2)
            iast_ch4_list.append(q_ch4)
            x_co2_list.append(x_co2)
        else:
            iast_co2_list.append(np.nan)
            iast_ch4_list.append(np.nan)
            x_co2_list.append(np.nan)
            failed_count += 1
    #     q = pyiast.iast(
    #         [pp_co2, pp_ch4],
    #         [isotherm_co2, isotherm_ch4],
    #         adsorbed_mole_fraction_guess=[y_co2, y_ch4],
    #         verboseflag=True
    #     )
    #     q_co2, q_ch4 = q[0], q[1]
 
    #     # Adsorbed phase mole fraction
    #     total_q = q_co2 + q_ch4
    #     x_co2   = q_co2 / total_q if total_q > 0 else np.nan
 
    #     iast_co2_list.append(q_co2)
    #     iast_ch4_list.append(q_ch4)
    #     x_co2_list.append(x_co2)

    except Exception as e:
        print(f"P={row['Pressure(bar)']:.4g}  ERROR: {e}")
    # except Exception:
    #     iast_co2_list.append(np.nan)
    #     iast_ch4_list.append(np.nan)
    #     x_co2_list.append(np.nan)
 
print("IAST calculations complete.")

#====== Write IAST Results to File ======
df_iast = pd.DataFrame({
    'Pressure(Pa)'      : df['Pressure(Pa)'],
    'Pressure(bar)'     : df['Pressure(bar)'],
    'Temperature(K)'    : df['Temperature(K)'],
    'CO2_mole_frac'     : df['CO2_mole_frac'],
    'CH4_mole_frac'     : df['CH4_mole_frac'],
    'GCMC_CO2_uptake'   : df['CO2_uptake'],
    'GCMC_CH4_uptake'   : df['CH4_uptake'],
    'IAST_CO2_uptake'   : iast_co2_list,
    'IAST_CH4_uptake'   : iast_ch4_list,
    'IAST_x_CO2'        : x_co2_list,
})
 
df_iast.to_csv('iast_pred_px.csv', index=False)
print("\nIAST predictions saved to: iast_pred_px.csv")

# Calculate Metrics
for comp, gcmc_col, iast_col in [
    ('CO2', 'GCMC_CO2_uptake', 'IAST_CO2_uptake'),
    ('CH4', 'GCMC_CH4_uptake', 'IAST_CH4_uptake'),
]:
    valid = df_iast[[gcmc_col, iast_col]].dropna()
    valid = valid[valid[gcmc_col] != 0]   # exclude zero GCMC values for MAPE
 
    gcmc = valid[gcmc_col].values
    iast = valid[iast_col].values
 
    mae  = np.abs(gcmc - iast).mean()
    rmse = np.sqrt(((gcmc - iast) ** 2).mean())
    mape = (np.abs((gcmc - iast) / gcmc) * 100).mean()
    r2   = 1 - np.sum((gcmc - iast) ** 2) / np.sum((gcmc - gcmc.mean()) ** 2)
    bias = (iast - gcmc).mean()
 
    print(f"\n{comp} ({len(valid)} valid points):")
    print(f"  MAE   = {mae:.6f} mmol/g")
    print(f"  RMSE  = {rmse:.6f} mmol/g")
    print(f"  MAPE  = {mape:.3f} %")
    print(f"  R²    = {r2:.6f}")
    print(f"  Bias  = {bias:.6f} mmol/g  ({'over' if bias > 0 else 'under'}-prediction)")
 
# Save metrics to a separate summary file
metrics_rows = []
for comp, gcmc_col, iast_col in [
    ('CO2', 'GCMC_CO2_uptake', 'IAST_CO2_uptake'),
    ('CH4', 'GCMC_CH4_uptake', 'IAST_CH4_uptake'),
]:
    valid = df_iast[[gcmc_col, iast_col]].dropna()
    valid = valid[valid[gcmc_col] != 0]
    gcmc  = valid[gcmc_col].values
    iast  = valid[iast_col].values
    metrics_rows.append({
        'Component' : comp,
        'N_points'  : len(valid),
        'MAE'       : np.abs(gcmc - iast).mean(),
        'RMSE'      : np.sqrt(((gcmc - iast) ** 2).mean()),
        'MAPE(%)'   : (np.abs((gcmc - iast) / gcmc) * 100).mean(),
        'R2'        : 1 - np.sum((gcmc - iast) ** 2) / np.sum((gcmc - gcmc.mean()) ** 2),
        'Bias'      : (iast - gcmc).mean(),
    })
 
pd.DataFrame(metrics_rows).to_csv('iast_metrics_px.csv', index=False)
print("\nMetrics saved to: iast_metrics_px.csv")