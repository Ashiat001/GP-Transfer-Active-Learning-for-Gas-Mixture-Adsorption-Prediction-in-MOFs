import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#======= Read in the GCMC, TAL Prediction and IAST Prediction data ========
# P-X
df = pd.read_excel('complete_Krish_1.xlsx')
df2 = pd.read_csv('iast_pred_px_1.csv')
TAL_px = pd.read_csv('Pred_TAL.csv')

# P-X-T
df3 = pd.read_csv('complete_Krish.csv')
df4 = pd.read_csv('iast_pred_pxt.csv')
TAL_pxt = pd.read_csv('Pred_TAL_pxt.csv')

#======= Define Plot Parameters and Plot Data ========
mole_frac = [0.06, 0.1, 0.2, 0.6, 0.8, 0.9]

# P-X
row_labels = ["(a)", "(b)", "(c)"]
row_y_positions = [0.915, 0.609, 0.30]

fig, axes = plt.subplots(figsize = (15, 18), nrows=3, ncols=2)

for ax, i in zip(axes.flat, mole_frac):
    # GCMC
    press_gcmc = df[df['CO2_mole_frac'] == i]['Pressure(Pa)'] / 1e5
    upt_ch_gcmc = df[df['CO2_mole_frac'] == i]['CH4_uptake']
    upt_co_gcmc = df[df['CO2_mole_frac'] == i]['CO2_uptake']
    error_ch = df[df['CO2_mole_frac'] == i]['CH4_Error']
    error_ch = 2 * error_ch
    error_co = df[df['CO2_mole_frac'] == i]['CO2_Error']
    error_co = 2 * error_co

    # IAST
    press_iast = df2[df2['CO2_mole_frac'] == i]['Pressure(Pa)'] / 1e5
    upt_ch_iast = df2[df2['CO2_mole_frac'] == i]['IAST_CH4_uptake']
    upt_co_iast = df2[df2['CO2_mole_frac'] == i]['IAST_CO2_uptake']

    # TAL
    press_tal = TAL_px[TAL_px['CO2_mole_fraction'] == i]['X Test']
    upt_ch_tal = TAL_px[TAL_px['CO2_mole_fraction'] == i]['CH4_Pred']
    upt_co_tal = TAL_px[TAL_px['CO2_mole_fraction'] == i]['CO2_Pred']

    # AL
    # press_al = gf[gf['CO2_mole_fraction'] == i]['X Test']
    # upt_ch_al = gf[gf['CO2_mole_fraction'] == i]['CH4_Pred']
    # upt_co_al = gf[gf['CO2_mole_fraction'] == i]['CO2_Pred']

    # CH4
    ax.plot(press_gcmc, upt_ch_gcmc, marker='o', markersize='6', markeredgecolor='darkmagenta', linestyle=':', linewidth='4', color='purple', label="$CH_{4}$ (GCMC)")
    ax.fill_between(press_gcmc, (upt_ch_gcmc - error_ch), (upt_ch_gcmc + error_ch),color='gray', alpha=0.4)
    ax.plot(press_iast, upt_ch_iast, marker='s', markersize='6', markeredgecolor='mediumvioletred', linestyle=':', linewidth='4', color='pink', label="$CH_{4}$ (IAST)")
    ax.plot(press_tal, upt_ch_tal, marker='p', markersize='6', markeredgecolor='r', linestyle=':', linewidth='4', color='crimson', label="$CH_{4}$ (TAL)")
    # ax.plot(press_al, upt_ch_al, marker='p', markersize='6', markeredgecolor='r', linestyle=':', linewidth='4', color='red', label="$CH_{4}$ (AL)")
    
    # CO2
    ax.plot(press_gcmc, upt_co_gcmc, marker='^', markersize='6', markeredgecolor='b', linestyle=':', linewidth='4', color='steelblue', label="$CO_{2}$ (GCMC)")
    ax.plot(press_iast, upt_co_iast, marker='P', markersize='6', markeredgecolor='g', linestyle=':', linewidth='4', color='mediumseagreen', label="$CO_{2}$ (IAST)")
    ax.plot(press_tal, upt_co_tal, marker='D', markersize='6', markeredgecolor='c', linestyle=':', linewidth='4', color='cyan', label="$CO_{2}$ (TAL)")
    # ax.plot(press_al, upt_co_al, marker='D', markersize='6', markeredgecolor='r', linestyle=':', linewidth='4', color='orange', label="$CO_{2}$ (AL)")
    ax.fill_between(press_gcmc, (upt_co_gcmc - error_co), (upt_co_gcmc + error_co),color='gray', alpha=0.4,label="95% Confidence")
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title("Adsorption Isotherms Comparison at $X_{CO_{2}}$ = " + f"{i}", fontsize = 15)
    ax.set_xlabel("Pressure (bar)", fontsize = 15, fontweight="bold")
    ax.set_ylabel("Adsorption Uptake (mg/g)", fontsize = 15, fontweight="bold")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(13)
        tick.set_fontweight("bold")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=15, frameon=False, bbox_to_anchor=(0.5, 0.96))
# Leave room for legend at top and row labels at left
fig.subplots_adjust(left=0.075,right=0.98,bottom=0.055,top=0.88,wspace=0.25,hspace=0.35)
for label, y in zip(row_labels, row_y_positions):
    fig.text(0.035,y,label,fontsize=15,fontweight="bold",ha="left",va="center")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('Adsorption_Isotherm_Comp_px_1.png', dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.savefig('Adsorption_Isotherm_Comp_px.pdf', dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.show()

#======= Define Plot Parameters and Plot Data ========
mole_fract = [0.02, 0.4, 0.98]

# P-X
panel_labels = ["(a)", "(b)", "(c)"]

fig, axes = plt.subplots(figsize = (20, 7), nrows=1, ncols=3)

for ax, i in zip(axes.flat, mole_fract):
    # GCMC
    press_gcmc = df[df['CO2_mole_frac'] == i]['Pressure(Pa)'] / 1e5
    upt_ch_gcmc = df[df['CO2_mole_frac'] == i]['CH4_uptake']
    upt_co_gcmc = df[df['CO2_mole_frac'] == i]['CO2_uptake']
    error_ch = df[df['CO2_mole_frac'] == i]['CH4_Error']
    error_ch = 2 * error_ch
    error_co = df[df['CO2_mole_frac'] == i]['CO2_Error']
    error_co = 2 * error_co

    # IAST
    press_iast = df2[df2['CO2_mole_frac'] == i]['Pressure(Pa)'] / 1e5
    upt_ch_iast = df2[df2['CO2_mole_frac'] == i]['IAST_CH4_uptake']
    upt_co_iast = df2[df2['CO2_mole_frac'] == i]['IAST_CO2_uptake']

    # TAL
    press_tal = TAL_px[TAL_px['CO2_mole_fraction'] == i]['X Test']
    upt_ch_tal = TAL_px[TAL_px['CO2_mole_fraction'] == i]['CH4_Pred']
    upt_co_tal = TAL_px[TAL_px['CO2_mole_fraction'] == i]['CO2_Pred']

    # AL
    # press_al = gf[gf['CO2_mole_fraction'] == i]['X Test']
    # upt_ch_al = gf[gf['CO2_mole_fraction'] == i]['CH4_Pred']
    # upt_co_al = gf[gf['CO2_mole_fraction'] == i]['CO2_Pred']

    # CH4
    ax.plot(press_gcmc, upt_ch_gcmc, marker='o', markersize='6', markeredgecolor='darkmagenta', linestyle=':', linewidth='4', color='purple', label="$CH_{4}$ (GCMC)")
    ax.fill_between(press_gcmc, (upt_ch_gcmc - error_ch), (upt_ch_gcmc + error_ch),color='gray', alpha=0.4)
    ax.plot(press_iast, upt_ch_iast, marker='s', markersize='6', markeredgecolor='mediumvioletred', linestyle=':', linewidth='4', color='pink', label="$CH_{4}$ (IAST)")
    ax.plot(press_tal, upt_ch_tal, marker='p', markersize='6', markeredgecolor='r', linestyle=':', linewidth='4', color='crimson', label="$CH_{4}$ (TAL)")
    # ax.plot(press_al, upt_ch_al, marker='p', markersize='6', markeredgecolor='r', linestyle=':', linewidth='4', color='red', label="$CH_{4}$ (AL)")
    
    # CO2
    ax.plot(press_gcmc, upt_co_gcmc, marker='^', markersize='6', markeredgecolor='b', linestyle=':', linewidth='4', color='steelblue', label="$CO_{2}$ (GCMC)")
    ax.plot(press_iast, upt_co_iast, marker='P', markersize='6', markeredgecolor='g', linestyle=':', linewidth='4', color='mediumseagreen', label="$CO_{2}$ (IAST)")
    ax.plot(press_tal, upt_co_tal, marker='D', markersize='6', markeredgecolor='c', linestyle=':', linewidth='4', color='cyan', label="$CO_{2}$ (TAL)")
    # ax.plot(press_al, upt_co_al, marker='D', markersize='6', markeredgecolor='r', linestyle=':', linewidth='4', color='orange', label="$CO_{2}$ (AL)")
    ax.fill_between(press_gcmc, (upt_co_gcmc - error_co), (upt_co_gcmc + error_co),color='gray', alpha=0.4,label="95% Confidence")
    
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.set_title("Adsorption Isotherms Comparison at $X_{CO_{2}}$ = " + f"{i}", fontsize = 17)
    ax.set_xlabel("Pressure (bar)", fontsize = 15, fontweight="bold")
    ax.set_ylabel("Adsorption Uptake (mg/g)", fontsize = 15, fontweight="bold")
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(13)
        tick.set_fontweight("bold")

for i, ax in enumerate(axes.flat):
    ax.text(-0.10,1.04,panel_labels[i],transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=17, frameon=False, bbox_to_anchor=(0.5, 0.99))
fig.subplots_adjust(left=0.075,right=0.98,bottom=0.12,top=0.78,wspace=0.25)
plt.tight_layout(rect=[0, 0, 1, 0.91])
plt.savefig('Adsorption_Isotherm_Comp_px_2.png', dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.savefig('Adsorption_Isotherm_Comp_px_2.pdf', dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
plt.show()

# P-X-T
temp = [200, 240, 300, 320, 360, 400]
mole_fraction = [0.02, 0.116, 0.212, 0.404, 0.5, 0.596, 0.788, 0.884, 0.98]

for i in mole_fraction:
    fig, axes = plt.subplots(figsize = (20, 13), nrows=2, ncols=3)
    for ax, j in zip(axes.flat, temp):
        # GCMC
        press_gcmc_pxt = df3[(df3['Temperature(K)'] == j) & (df3['CO2_mole_frac'] == i)]['Pressure(Pa)'] / 1e5
        upt_ch_gcmc_pxt = df3[(df3['Temperature(K)'] == j) & (df3['CO2_mole_frac'] == i)]['CH4_uptake']
        upt_co_gcmc_pxt = df3[(df3['Temperature(K)'] == j) & (df3['CO2_mole_frac'] == i)]['CO2_uptake']
        error_ch_pxt = df3[(df3['Temperature(K)'] == j) & (df3['CO2_mole_frac'] == i)]['CH4_Error']
        error_ch_pxt = 2 * error_ch_pxt
        error_co_pxt = df3[(df3['Temperature(K)'] == j) & (df3['CO2_mole_frac'] == i)]['CO2_Error']
        error_co_pxt = 2 * error_co_pxt

        # IAST
        press_iast_pxt = df4[(df4['Temperature(K)'] == j) & (df4['CO2_mole_frac'] == i)]['Pressure(Pa)'] / 1e5
        upt_ch_iast_pxt = df4[(df4['Temperature(K)'] == j) & (df4['CO2_mole_frac'] == i)]['IAST_CH4_uptake']
        upt_co_iast_pxt = df4[(df4['Temperature(K)'] == j) & (df4['CO2_mole_frac'] == i)]['IAST_CO2_uptake']

        # TAL
        press_tal_pxt = TAL_pxt[(TAL_pxt['Temperature'] == j) & (TAL_pxt['CO2_mole_fraction'] == i)]['X Test']
        upt_ch_tal_pxt = TAL_pxt[(TAL_pxt['Temperature'] == j) & (TAL_pxt['CO2_mole_fraction'] == i)]['CH4_Pred']
        upt_co_tal_pxt = TAL_pxt[(TAL_pxt['Temperature'] == j) & (TAL_pxt['CO2_mole_fraction'] == i)]['CO2_Pred']

        # CH4
        ax.plot(press_gcmc_pxt, upt_ch_gcmc_pxt, marker='o', markersize='6', markeredgecolor='darkmagenta', linestyle=':', linewidth='4', color='purple', label="$CH_{4}$ (GCMC)")
        ax.fill_between(press_gcmc_pxt, (upt_ch_gcmc_pxt - error_ch_pxt), (upt_ch_gcmc_pxt + error_ch_pxt),color='gray', alpha=0.4)
        ax.plot(press_iast_pxt, upt_ch_iast_pxt, marker='s', markersize='6', markeredgecolor='mediumvioletred', linestyle=':', linewidth='4', color='pink', label="$CH_{4}$ (IAST)")
        ax.plot(press_tal_pxt, upt_ch_tal_pxt, marker='p', markersize='6', markeredgecolor='r', linestyle=':', linewidth='4', color='crimson', label="$CH_{4}$ (TAL)")
        
        # CO2
        ax.plot(press_gcmc_pxt, upt_co_gcmc_pxt, marker='^', markersize='6', markeredgecolor='b', linestyle=':', linewidth='4', color='steelblue', label="$CO_{2}$ (GCMC)")
        ax.plot(press_iast_pxt, upt_co_iast_pxt, marker='P', markersize='6', markeredgecolor='g', linestyle=':', linewidth='4', color='mediumseagreen', label="$CO_{2}$ (IAST)")
        ax.plot(press_tal_pxt, upt_co_tal_pxt, marker='D', markersize='6', markeredgecolor='c', linestyle=':', linewidth='4', color='cyan', label="$CO_{2}$ (TAL)")
        ax.fill_between(press_gcmc_pxt, (upt_co_gcmc_pxt - error_co_pxt), (upt_co_gcmc_pxt + error_co_pxt),color='gray', alpha=0.4,label="95% Confidence")
        
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.set_title("Isotherms at $X_{CO_{2}}$ = " + f'{i}, ' + f'Temp = {j}', fontsize = 17)
        ax.set_xlabel("Pressure (bar)", fontsize = 15, fontweight="bold")
        ax.set_ylabel("Adsorption Uptake (mg/g)", fontsize = 15, fontweight="bold")
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            tick.set_fontsize(13)
            tick.set_fontweight("bold")
        
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=7, fontsize=17, frameon=False, bbox_to_anchor=(0.5, 0.96))
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'Adsorption_Isotherm_Comp_pxt_{i}.png', dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.savefig(f'Adsorption_Isotherm_Comp_pxt_{i}.pdf', dpi=600, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.show()

#======= Generate Residual Contour Plots ========
# Compute residual P-X
residual_ch = df['CH4_uptake'] - TAL_px['CH4_Pred']
residual_ch = np.abs(residual_ch)

residual_co = df['CO2_uptake'] - TAL_px['CO2_Pred']
residual_co = np.abs(residual_co)

vmin = min(residual_co.min(), residual_ch.min())
vmax = max(residual_co.max(), residual_ch.max())

df['P_BAR'] = df['Pressure(Pa)'] / 1e5

panel_labels = ["(d)", "(e)"]
fig, ax = plt.subplots(figsize=(11, 5), ncols=2)

contour_ch = ax[0].tricontourf(df['CH4_mole_frac'], df['P_BAR'], residual_ch, levels=30, cmap="YlGnBu", vmin = vmin, vmax = vmax)
ax[0].set_xlabel(r"$\mathbf{CH}_{\mathbf{4}}$ mole fraction", fontsize = 14, fontweight="bold")
ax[0].set_ylabel("Pressure (bar)", fontsize = 14, fontweight="bold")
ax[0].set_title(r"$\mathrm{CH}_{4}$ Prediction Residuals for P-X Space", fontsize = 14)
for tick in ax[0].get_xticklabels() + ax[0].get_yticklabels():
    tick.set_fontsize(13)
    tick.set_fontweight("bold")

contour_co = ax[1].tricontourf(df['CO2_mole_frac'], df['P_BAR'], residual_co, levels=30, cmap="YlGnBu", vmin = vmin, vmax = vmax)
ax[1].set_xlabel(r"$\mathbf{CO}_{\mathbf{2}}$ mole fraction", fontsize = 14, fontweight="bold")
ax[1].set_ylabel("Pressure (bar)", fontsize = 14, fontweight="bold")
ax[1].set_title(r"$\mathrm{CO}_{2}$ Prediction Residuals for P-X Space", fontsize = 14)
for tick in ax[1].get_xticklabels() + ax[1].get_yticklabels():
    tick.set_fontsize(13)
    tick.set_fontweight("bold")

cbar_ax = fig.add_axes([1.0, 0.15, 0.02, 0.75])
cbar = fig.colorbar(contour_co, cax = cbar_ax, label="Absolute Residual (mg/g)", shrink=0.9)
cbar.set_label("Absolute Residual (mg/g)", fontsize=14)
cbar.ax.tick_params(labelsize=12)

for i, axes in enumerate(ax.flat):
    axes.text(-0.15,1.08,panel_labels[i],transform=axes.transAxes,
        fontsize=13,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)
plt.tight_layout()
plt.savefig('Residual_px.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('Residual_px.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()

# Compute Residual P-X-T
font_ppt = {'family': 'sans-serif', 'color': 'black', 'size': 14}
# CH4
all_residuals_ch = []

for T in temp:
    residual_T = np.abs(df3[df3['Temperature(K)'] == T]['CH4_uptake'] - TAL_pxt[TAL_pxt['Temperature'] == T]['CH4_Pred'])

    all_residuals_ch.append(residual_T)

vmin_ch = 0
vmax_ch = np.max(np.concatenate(all_residuals_ch))
levels_ch = np.linspace(vmin_ch, vmax_ch, 31)

fig, axes = plt.subplots(figsize=(15, 10), ncols=3, nrows=2)

for ax, T in zip(axes.flat, temp):
    mole_frac_ch = df3[df3['Temperature(K)'] == T]['CH4_mole_frac']
    press_ch = df3[df3['Temperature(K)'] == T]['Pressure(Pa)'] / 1e5
    residual_ch_pxt = np.abs(df3[df3['Temperature(K)'] == T]['CH4_uptake'] - TAL_pxt[TAL_pxt['Temperature'] == T]['CH4_Pred'])

    contour_ch_pxt = ax.tricontourf(mole_frac_ch, press_ch, residual_ch_pxt, levels=levels_ch, cmap="YlGnBu", vmin =vmin_ch, vmax = vmax_ch)
    ax.set_xlabel(r"$\mathbf{CH}_{\mathbf{4}}$ mole fraction", fontdict = font_ppt, fontweight="bold")
    ax.set_ylabel("Pressure (bar)", fontdict = font_ppt, fontweight="bold")
    ax.set_title(f"$CH_{4}$ Prediction Residuals at Temp = {T} K", fontdict= font_ppt)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight("bold")
# fig.suptitle("$CH_{4}$ TAL Prediction Residuals for P-X-T Space", fontsize=15)
cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
cbar = fig.colorbar(contour_ch_pxt, cax = cbar_ax, label="Absolute Residual (mg/g)", shrink=0.9)
cbar.set_label("Absolute Residual (mg/g)", fontsize=15)
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('Residual_ch_pxt.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('Residual_ch_pxt.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()

# CO2
all_residuals_co = []

for T in temp:
    residual_T = np.abs(df3[df3['Temperature(K)'] == T]['CO2_uptake'] - TAL_pxt[TAL_pxt['Temperature'] == T]['CO2_Pred'])

    all_residuals_co.append(residual_T)

vmin_co = 0
vmax_co = np.max(np.concatenate(all_residuals_co))
levels_co = np.linspace(vmin_co, vmax_co, 31)

fig, axes = plt.subplots(figsize=(15, 10), ncols=3, nrows=2)

for ax, T in zip(axes.flat, temp):
    mole_frac_co = df3[df3['Temperature(K)'] == T]['CO2_mole_frac']
    press_co = df3[df3['Temperature(K)'] == T]['Pressure(Pa)'] / 1e5
    residual_co_pxt = np.abs(df3[df3['Temperature(K)'] == T]['CO2_uptake'] - TAL_pxt[TAL_pxt['Temperature'] == T]['CO2_Pred'])

    contour_co_pxt = ax.tricontourf(mole_frac_co, press_co, residual_co_pxt, levels=levels_co, cmap="YlGnBu", vmin = vmin_co, vmax = vmax_co)
    ax.set_xlabel(r"$\mathbf{CO}_{\mathbf{2}}$ mole fraction", fontdict = font_ppt, fontweight="bold")
    ax.set_ylabel("Pressure (bar)", fontdict = font_ppt, fontweight="bold")
    ax.set_title(f"$CO_{2}$ Prediction Residuals at Temp = {T} K", fontdict= font_ppt)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(12)
        tick.set_fontweight("bold")
# fig.suptitle("$CO_{2}$ TAL Prediction Residuals for P-X-T Space", fontsize=15)
cbar_ax = fig.add_axes([1.02, 0.15, 0.02, 0.7])
cbar = fig.colorbar(contour_co_pxt, cax = cbar_ax, label="Absolute Residual (mg/g)", shrink=0.9,)
cbar.set_label("Absolute Residual (mg/g)", fontsize=15)
cbar.ax.tick_params(labelsize=14)
plt.tight_layout()
plt.savefig('Residual_co_pxt.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('Residual_co_pxt.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()

#======= Generate Residual Heatmap Plots =======
# P-X
df['CO2_residual'] = np.abs(df['CO2_uptake'] - TAL_px['CO2_Pred'])
df['CH4_residual'] = np.abs(df['CH4_uptake'] - TAL_px['CH4_Pred'])

heatmap_data_ch = df.pivot_table(index="P_BAR", columns="CH4_mole_frac", values="CH4_residual")
heatmap_data_co = df.pivot_table(index="P_BAR", columns="CO2_mole_frac", values="CO2_residual")

fig, ax = plt.subplots(figsize=(11, 5), ncols=2)

ax[0].imshow(heatmap_data_ch, aspect="auto", origin="lower", cmap="coolwarm",
    extent=[
        heatmap_data_ch.columns.min(),
        heatmap_data_ch.columns.max(),
        heatmap_data_ch.index.min(),
        heatmap_data_ch.index.max()
    ])
ax[0].set_xlabel("CH4 mole fraction")
ax[0].set_ylabel("Pressure (bar)")
ax[0].set_title(f"CH4 GP Prediction Residuals at 300 K")
# plt.colorbar(label="Residual: actual - predicted")

ax[1].imshow(heatmap_data_co, aspect="auto", origin="lower", cmap="coolwarm",
    extent=[
        heatmap_data_co.columns.min(),
        heatmap_data_co.columns.max(),
        heatmap_data_co.index.min(),
        heatmap_data_co.index.max()
    ])
ax[1].set_xlabel("CO2 mole fraction")
ax[1].set_ylabel("Pressure (bar)")
ax[1].set_title(f"CO2 GP Prediction Residuals at 300 K")
# plt.colorbar(label="Residual: actual - predicted")
plt.tight_layout()
plt.show()



