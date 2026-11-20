import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pyiast
import warnings
warnings.filterwarnings('ignore')
import os

# inputs: the GCMC data in Datasets/ (outputs are written to the folder you run this script from)
DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../Datasets/Cu-BTC/Xe-Kr'))
 
#====== Read in Pure and Mixture Dataset =======
df = pd.read_csv(os.path.join(DATA, 'Mixture_px.csv'))
df2 = pd.read_csv(os.path.join(DATA, 'Pure_Xe_data.csv'))
df3 = pd.read_csv(os.path.join(DATA, 'Pure_Kr_data.csv')) 
 
# Convert pressure from Pa to bar 
df['Pressure(bar)'] = df['Pressure(Pa)'] / 1e5
df2['Pressure [bar]'] = df2['Pressure [Pa]'] / 1e5
df3['Pressure [bar]'] = df3['Pressure [Pa]'] / 1e5
 
#====== Fit Pure Component Isotherms ======
print("Fitting Kr isotherm...")
isotherm_Kr = pyiast.InterpolatorIsotherm(df3,
    loading_key='Uptake [mg/g]',
    pressure_key='Pressure [bar]',
    fill_value=df3['Uptake [mg/g]'].max()
)
 
print("\nFitting Xe isotherm...")
isotherm_Xe = pyiast.InterpolatorIsotherm(
    df2,
    loading_key='Uptake [mg/g]',
    pressure_key='Pressure [bar]',
    fill_value=df2['Uptake [mg/g]'].max()
)

# Pre-fetch the lowest data point for Henry's law fallback
Kr_pp_min = df3['Pressure [bar]'].iloc[0]
Kr_ld_min = df3['Uptake [mg/g]'].iloc[0]
Xe_pp_min = df2['Pressure [bar]'].iloc[0]
Xe_ld_min = df2['Uptake [mg/g]'].iloc[0]
 
KH_Kr = Kr_ld_min / Kr_pp_min if Kr_pp_min > 0 else 0.0
KH_Xe = Xe_ld_min / Xe_pp_min if Xe_pp_min > 0 else 0.0

#====== Implement IAST for Full Pressure and Mole Fraction Range ======

iast_Kr_list   = []
iast_Xe_list   = []
x_Kr_list      = []
failed_count     = 0
henry_count      = 0
 
for _, row in df.iterrows():
    p_total = row['Pressure(bar)']
    y_Kr   = row['Kr_mole_frac']
    y_Xe   = row['Xe_mole_frac']
    
    pp_Kr  = y_Kr * p_total
    pp_Xe  = y_Xe * p_total
 
    # Henry's law fallback for partial pressures below the isotherm data range
    if pp_Kr < Kr_pp_min or pp_Xe < Xe_pp_min:
        q_Kr = KH_Kr * pp_Kr
        q_Xe = KH_Xe * pp_Xe
        total = q_Kr + q_Xe
        x_Kr = q_Kr / total if total > 0 else np.nan
        iast_Kr_list.append(q_Kr)
        iast_Xe_list.append(q_Xe)
        x_Kr_list.append(x_Kr)
        henry_count += 1
        continue
    try:
        q = pyiast.iast(
            [pp_Kr, pp_Xe],
            [isotherm_Kr, isotherm_Xe],
        )
        q_Kr, q_Xe = q[0], q[1]
 
        # Adsorbed phase mole fraction
        total_q = q_Kr + q_Xe
        x_Kr   = q_Kr / total_q if total_q > 0 else np.nan
 
        iast_Kr_list.append(q_Kr)
        iast_Xe_list.append(q_Xe)
        x_Kr_list.append(x_Kr)
 
    except Exception:
        iast_Kr_list.append(np.nan)
        iast_Xe_list.append(np.nan)
        x_Kr_list.append(np.nan)
 
print("IAST calculations complete.")

#====== Write IAST Results to File ======
df_iast = pd.DataFrame({
    'Pressure(Pa)'      : df['Pressure(Pa)'],
    'Pressure(bar)'     : df['Pressure(bar)'],
    'Temperature(K)'    : df['Temperature(K)'],
    'Kr_mole_frac'     : df['Kr_mole_frac'],
    'Xe_mole_frac'     : df['Xe_mole_frac'],
    'GCMC_Kr_uptake'   : df['Kr_uptake'],
    'GCMC_Xe_uptake'   : df['Xe_uptake'],
    'IAST_Kr_uptake'   : iast_Kr_list,
    'IAST_Xe_uptake'   : iast_Xe_list,
    'IAST_x_Kr'        : x_Kr_list,
})
 
df_iast.to_csv('iast_pred_px.csv', index=False)
print("\nIAST predictions saved to: iast_pred_px.csv")

# Calculate Metrics
for comp, gcmc_col, iast_col in [
    ('Kr', 'GCMC_Kr_uptake', 'IAST_Kr_uptake'),
    ('Xe', 'GCMC_Xe_uptake', 'IAST_Xe_uptake'),
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
    ('Kr', 'GCMC_Kr_uptake', 'IAST_Kr_uptake'),
    ('Xe', 'GCMC_Xe_uptake', 'IAST_Xe_uptake'),
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