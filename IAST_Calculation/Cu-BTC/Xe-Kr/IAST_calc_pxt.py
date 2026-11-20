import numpy as np
import pandas as pd
import pyiast
import warnings
warnings.filterwarnings('ignore')
import os

# inputs: the GCMC data in Datasets/ (outputs are written to the folder you run this script from)
DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../Datasets/Cu-BTC/Xe-Kr'))
 
#====== Read in Pure and Mixture Dataset =======
df_mix = pd.read_csv(os.path.join(DATA, 'Mixture_pxt.csv')).rename(columns = {'P-bar': 'P_BAR'})    # pressure in bar
 
#======= Extract Pseudo Pure Component Data from the Mixture Data ========
#   Pure Kr proxy → rows where Kr_mole_frac is maximum (0.98)
#   Pure Xe proxy → rows where Xe_mole_frac is maximum (i.e. Kr_mole_frac minimum = 0.02)

Y_Kr_PURE = df_mix['Kr_mole_frac'].max()  
Y_Xe_PURE = df_mix['Kr_mole_frac'].min()   
 
#======== Fit Pure Component Isotherm Per Temperature =======
isotherms_Kr = {}
isotherms_Xe = {}
 
print("\nFitting isotherms per temperature...")

temperatures = sorted(df_mix['Temperature(K)'].unique())
 
for T in temperatures:
    df_T = df_mix[df_mix['Temperature(K)'] == T].copy()
 
    # --- Pure Kr proxy isotherm ---
    Kr_rows = df_T[df_T['Kr_mole_frac'] == Y_Kr_PURE].sort_values('P_BAR')
    df_Kr_iso = pd.DataFrame({
        'pressure': (Kr_rows['Kr_mole_frac'] * Kr_rows['P_BAR']).values,
        'loading' : Kr_rows['Kr_uptake'].values
    }).dropna()
    df_Kr_iso = df_Kr_iso[
        (df_Kr_iso['pressure'] > 0) & (df_Kr_iso['loading'] > 0)
    ].sort_values('pressure')
 
    # --- Pure Xe proxy isotherm ---
    Xe_rows = df_T[df_T['Kr_mole_frac'] == Y_Xe_PURE].sort_values('P_BAR')
    df_Xe_iso = pd.DataFrame({
        'pressure': (Xe_rows['Xe_mole_frac'] * Xe_rows['P_BAR']).values,
        'loading' : Xe_rows['Xe_uptake'].values
    }).dropna()
    df_Xe_iso = df_Xe_iso[
        (df_Xe_iso['pressure'] > 0) & (df_Xe_iso['loading'] > 0)
    ].sort_values('pressure')
 
    if df_Kr_iso.empty or df_Xe_iso.empty:
        print(f"  T={T}K : WARNING — not enough data to fit, skipping.")
        continue
 
    # --- Fit Kr ---
    try:
        isotherms_Kr[T] = pyiast.InterpolatorIsotherm(
            df_Kr_iso,
            loading_key='loading',
            pressure_key='pressure',
            fill_value=df_Kr_iso['loading'].max()
        )
    except Exception as e:
        print(f"  T={T}K Kr fit failed: {e}")
 
    # --- Fit Xe ---
    try:
        isotherms_Xe[T] = pyiast.InterpolatorIsotherm(
            df_Xe_iso,
            loading_key='loading',
            pressure_key='pressure',
            fill_value=df_Xe_iso['loading'].max()
        )
    except Exception as e:
        print(f"  T={T}K Xe fit failed: {e}")
 
    print(f"  T={T}K : fitted  "
          f"(Kr pts={len(df_Kr_iso)}, Xe pts={len(df_Xe_iso)})")
 
fitted_temps = sorted(set(isotherms_Kr.keys()) & set(isotherms_Xe.keys()))
print(f"\nIsotherms ready for {len(fitted_temps)} temperatures: {fitted_temps}")
 
#======== Run IAST Calculations Over the P-X-T Space =========
print("\nRunning IAST over all mixture conditions...")
 
iast_Kr_list    = []
iast_Xe_list    = []
x_Kr_list       = []
 
for _, row in df_mix.iterrows():
    T       = row['Temperature(K)']
    p_total = row['P_BAR']
    y_Kr   = row['Kr_mole_frac']
    y_Xe   = row['Xe_mole_frac']
 
    if T not in isotherms_Kr or T not in isotherms_Xe:
        iast_Kr_list.append(np.nan)
        iast_Xe_list.append(np.nan)
        x_Kr_list.append(np.nan)
        continue

    pp_Kr = y_Kr * p_total
    pp_Xe = y_Xe * p_total
 
    # Skip conditions where partial pressure is below the isotherm data range
    # (interpolator cannot extrapolate reliably below the lowest data point)
    Kr_pp_min = isotherms_Kr[T].df[isotherms_Kr[T].pressure_key].min()
    Xe_pp_min = isotherms_Xe[T].df[isotherms_Xe[T].pressure_key].min()
 
    if pp_Kr < Kr_pp_min or pp_Xe < Xe_pp_min:
        # Use Henry's law approximation at very low pressures:
        # q_i ≈ K_H_i * p_i  where K_H = loading/pressure at lowest data point
        Kr_pp_low = isotherms_Kr[T].df[isotherms_Kr[T].pressure_key].iloc[0]
        Kr_ld_low = isotherms_Kr[T].df[isotherms_Kr[T].loading_key].iloc[0]
        Xe_pp_low = isotherms_Xe[T].df[isotherms_Xe[T].pressure_key].iloc[0]
        Xe_ld_low = isotherms_Xe[T].df[isotherms_Xe[T].loading_key].iloc[0]
 
        KH_Kr = Kr_ld_low / Kr_pp_low if Kr_pp_low > 0 else 0
        KH_Xe = Xe_ld_low / Xe_pp_low if Xe_pp_low > 0 else 0
 
        q_Kr = KH_Kr * pp_Kr
        q_Xe = KH_Xe * pp_Xe
 
        total = q_Kr + q_Xe
        x_Kr = q_Kr / total if total > 0 else np.nan
 
        iast_Kr_list.append(q_Kr)
        iast_Xe_list.append(q_Xe)
        x_Kr_list.append(x_Kr)
        continue
 
    try:
        q = pyiast.iast(
            [pp_Kr, pp_Xe],
            [isotherms_Kr[T], isotherms_Xe[T]],
        )
        q_Kr, q_Xe = q[0], q[1]
 
        total = q_Kr + q_Xe
        x_Kr = q_Kr / total if total > 0 else np.nan
 
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
    'Pressure(Pa)'      : df_mix['Pressure(Pa)'],
    'Pressure(bar)'     : df_mix['P_BAR'],
    'Temperature(K)'    : df_mix['Temperature(K)'],
    'Kr_mole_frac'     : df_mix['Kr_mole_frac'],
    'Xe_mole_frac'     : df_mix['Xe_mole_frac'],
    'GCMC_Kr_uptake'   : df_mix['Kr_uptake'],
    'GCMC_Xe_uptake'   : df_mix['Xe_uptake'],
    'IAST_Kr_uptake'   : iast_Kr_list,
    'IAST_Xe_uptake'   : iast_Xe_list,
    'IAST_x_Kr'        : x_Kr_list,
})
 
df_iast.to_csv('iast_pred_pxt.csv', index=False)
print("\nIAST predictions saved to: iast_pred_pxt.csv")
 
# Calculate Metrics

def compute_metrics(gcmc, iast):
    err  = iast - gcmc
    mask = gcmc != 0
    return {
        'N_points' : len(gcmc),
        'MAE'      : np.abs(err).mean(),
        'RMSE'     : np.sqrt((err ** 2).mean()),
        'MAPE(%)'  : (np.abs(err[mask] / gcmc[mask]) * 100).mean() if mask.sum() > 0 else np.nan,
        'R2'       : 1 - np.sum(err**2) / np.sum((gcmc - gcmc.mean())**2),
        'Bias'     : err.mean(),
    }
 
metrics_rows = []
 
for comp, gcmc_col, iast_col in [
    ('Kr', 'GCMC_Kr_uptake', 'IAST_Kr_uptake'),
    ('Xe', 'GCMC_Xe_uptake', 'IAST_Xe_uptake'),
]:
    print(f"\n{'─'*55}")
    print(f"  {comp}")
    print(f"{'─'*55}")
 
    # Overall
    valid = df_iast[[gcmc_col, iast_col, 'Temperature(K)']].dropna()
    m = compute_metrics(valid[gcmc_col].values, valid[iast_col].values)
    print(f"\n  Overall ({m['N_points']} points):")
    print(f"    MAE    = {m['MAE']:.6f} mmol/g")
    print(f"    RMSE   = {m['RMSE']:.6f} mmol/g")
    print(f"    MAPE   = {m['MAPE(%)']:.3f} %")
    print(f"    R²     = {m['R2']:.6f}")
    print(f"    Bias   = {m['Bias']:.6f} mmol/g  ({'over' if m['Bias'] > 0 else 'under'}-prediction)")
    metrics_rows.append({'Component': comp, 'Temperature(K)': 'Overall', **m})
 
    # Per temperature
    print(f"\n  Per temperature:")
    for T in fitted_temps:
        sub = valid[valid['Temperature(K)'] == T]
        if sub.empty:
            continue
        mt = compute_metrics(sub[gcmc_col].values, sub[iast_col].values)
        print(f"    T={T}K  N={mt['N_points']:4d}  "
              f"MAE={mt['MAE']:.5f}  RMSE={mt['RMSE']:.5f}  "
              f"MAPE={mt['MAPE(%)']:7.3f}%  R²={mt['R2']:.5f}  "
              f"Bias={mt['Bias']:+.5f}")
        metrics_rows.append({'Component': comp, 'Temperature(K)': T, **mt})
 
pd.DataFrame(metrics_rows).to_csv('iast_metrics_pxt.csv', index=False)
print("\nMetrics saved to: iast_metrics_pxt.csv")