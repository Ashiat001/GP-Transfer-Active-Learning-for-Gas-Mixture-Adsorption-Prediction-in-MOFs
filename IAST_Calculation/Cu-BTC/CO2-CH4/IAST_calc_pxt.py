import numpy as np
import pandas as pd
import pyiast
import warnings
warnings.filterwarnings('ignore')
import os

# inputs: the GCMC data in Datasets/ (outputs are written to the folder you run this script from)
DATA = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../Datasets/Cu-BTC/CO2-CH4'))
 
#====== Read in Pure and Mixture Dataset =======
df_mix = pd.read_csv(os.path.join(DATA, 'Mixture_pxt.csv'))
 
#======= Extract Pseudo Pure Component Data from the Mixture Data ========
#   Pure CO2 proxy → rows where CO2_mole_frac is maximum (0.98)
#   Pure CH4 proxy → rows where CH4_mole_frac is maximum (i.e. CO2_mole_frac minimum = 0.02)

Y_CO2_PURE = df_mix['CO2_mole_frac'].max()  
Y_CH4_PURE = df_mix['CO2_mole_frac'].min()   
 
#======== Fit Pure Component Isotherm Per Temperature =======
isotherms_co2 = {}
isotherms_ch4 = {}
 
print("\nFitting isotherms per temperature...")

temperatures = sorted(df_mix['Temperature(K)'].unique())
 
for T in temperatures:
    df_T = df_mix[df_mix['Temperature(K)'] == T].copy()
 
    # --- Pure CO2 proxy isotherm ---
    co2_rows = df_T[df_T['CO2_mole_frac'] == Y_CO2_PURE].sort_values('P_BAR')
    df_co2_iso = pd.DataFrame({
        'pressure': (co2_rows['CO2_mole_frac'] * co2_rows['P_BAR']).values,
        'loading' : co2_rows['CO2_uptake'].values
    }).dropna()
    df_co2_iso = df_co2_iso[
        (df_co2_iso['pressure'] > 0) & (df_co2_iso['loading'] > 0)
    ].sort_values('pressure')
 
    # --- Pure CH4 proxy isotherm ---
    ch4_rows = df_T[df_T['CO2_mole_frac'] == Y_CH4_PURE].sort_values('P_BAR')
    df_ch4_iso = pd.DataFrame({
        'pressure': (ch4_rows['CH4_mole_frac'] * ch4_rows['P_BAR']).values,
        'loading' : ch4_rows['CH4_uptake'].values
    }).dropna()
    df_ch4_iso = df_ch4_iso[
        (df_ch4_iso['pressure'] > 0) & (df_ch4_iso['loading'] > 0)
    ].sort_values('pressure')
 
    if df_co2_iso.empty or df_ch4_iso.empty:
        print(f"  T={T}K : WARNING — not enough data to fit, skipping.")
        continue
 
    # --- Fit CO2 ---
    try:
        isotherms_co2[T] = pyiast.InterpolatorIsotherm(
            df_co2_iso,
            loading_key='loading',
            pressure_key='pressure',
            fill_value=df_co2_iso['loading'].max()
        )
    except Exception as e:
        print(f"  T={T}K CO2 fit failed: {e}")
 
    # --- Fit CH4 ---
    try:
        isotherms_ch4[T] = pyiast.InterpolatorIsotherm(
            df_ch4_iso,
            loading_key='loading',
            pressure_key='pressure',
            fill_value=df_ch4_iso['loading'].max()
        )
    except Exception as e:
        print(f"  T={T}K CH4 fit failed: {e}")
 
    print(f"  T={T}K : fitted  "
          f"(CO2 pts={len(df_co2_iso)}, CH4 pts={len(df_ch4_iso)})")
 
fitted_temps = sorted(set(isotherms_co2.keys()) & set(isotherms_ch4.keys()))
print(f"\nIsotherms ready for {len(fitted_temps)} temperatures: {fitted_temps}")
 
#======== Run IAST Calculations Over the P-X-T Space =========
print("\nRunning IAST over all mixture conditions...")
 
iast_co2_list    = []
iast_ch4_list    = []
x_co2_list       = []
 
for _, row in df_mix.iterrows():
    T       = row['Temperature(K)']
    p_total = row['P_BAR']
    y_co2   = row['CO2_mole_frac']
    y_ch4   = row['CH4_mole_frac']
 
    if T not in isotherms_co2 or T not in isotherms_ch4:
        iast_co2_list.append(np.nan)
        iast_ch4_list.append(np.nan)
        x_co2_list.append(np.nan)
        continue

    pp_co2 = y_co2 * p_total
    pp_ch4 = y_ch4 * p_total
 
    # Skip conditions where partial pressure is below the isotherm data range
    # (interpolator cannot extrapolate reliably below the lowest data point)
    co2_pp_min = isotherms_co2[T].df[isotherms_co2[T].pressure_key].min()
    ch4_pp_min = isotherms_ch4[T].df[isotherms_ch4[T].pressure_key].min()
 
    if pp_co2 < co2_pp_min or pp_ch4 < ch4_pp_min:
        # Use Henry's law approximation at very low pressures:
        # q_i ≈ K_H_i * p_i  where K_H = loading/pressure at lowest data point
        co2_pp_low = isotherms_co2[T].df[isotherms_co2[T].pressure_key].iloc[0]
        co2_ld_low = isotherms_co2[T].df[isotherms_co2[T].loading_key].iloc[0]
        ch4_pp_low = isotherms_ch4[T].df[isotherms_ch4[T].pressure_key].iloc[0]
        ch4_ld_low = isotherms_ch4[T].df[isotherms_ch4[T].loading_key].iloc[0]
 
        KH_co2 = co2_ld_low / co2_pp_low if co2_pp_low > 0 else 0
        KH_ch4 = ch4_ld_low / ch4_pp_low if ch4_pp_low > 0 else 0
 
        q_co2 = KH_co2 * pp_co2
        q_ch4 = KH_ch4 * pp_ch4
 
        total = q_co2 + q_ch4
        x_co2 = q_co2 / total if total > 0 else np.nan
 
        iast_co2_list.append(q_co2)
        iast_ch4_list.append(q_ch4)
        x_co2_list.append(x_co2)
        continue
 
    try:
        q = pyiast.iast(
            [pp_co2, pp_ch4],
            [isotherms_co2[T], isotherms_ch4[T]],
        )
        q_co2, q_ch4 = q[0], q[1]
 
        total = q_co2 + q_ch4
        x_co2 = q_co2 / total if total > 0 else np.nan
 
        iast_co2_list.append(q_co2)
        iast_ch4_list.append(q_ch4)
        x_co2_list.append(x_co2)
 
    except Exception:
        iast_co2_list.append(np.nan)
        iast_ch4_list.append(np.nan)
        x_co2_list.append(np.nan)
 
print("IAST calculations complete.")
 
#====== Write IAST Results to File ======
df_iast = pd.DataFrame({
    'Pressure(Pa)'      : df_mix['Pressure(Pa)'],
    'Pressure(bar)'     : df_mix['P_BAR'],
    'Temperature(K)'    : df_mix['Temperature(K)'],
    'CO2_mole_frac'     : df_mix['CO2_mole_frac'],
    'CH4_mole_frac'     : df_mix['CH4_mole_frac'],
    'GCMC_CO2_uptake'   : df_mix['CO2_uptake'],
    'GCMC_CH4_uptake'   : df_mix['CH4_uptake'],
    'IAST_CO2_uptake'   : iast_co2_list,
    'IAST_CH4_uptake'   : iast_ch4_list,
    'IAST_x_CO2'        : x_co2_list,
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
    ('CO2', 'GCMC_CO2_uptake', 'IAST_CO2_uptake'),
    ('CH4', 'GCMC_CH4_uptake', 'IAST_CH4_uptake'),
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