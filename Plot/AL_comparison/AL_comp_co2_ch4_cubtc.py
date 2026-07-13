import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#====== Read in Data Files ======
# P-X
# AL Data CH4
df = pd.read_csv('AL_ch_sampled points.csv')
df2 = pd.read_csv('AL_ch_MRE.csv')
df6 = pd.read_csv('AL_CH_PAC_px.csv')

# AL Data CO2
gf = pd.read_csv('AL_co_sampled points.csv')
gf2 = pd.read_csv('AL_co_MRE.csv')
gf6 = pd.read_csv('AL_CO_PAC_px.csv')

# TAL Data CH4
hf = pd.read_csv('TAL_CH_sampled points_px.csv')
hf2 = pd.read_csv('TAL_CH_MRE_px.csv')
hf6 = pd.read_csv('TAL_CH_PAC_px.csv')

# TAL Data CO2
kf = pd.read_csv('TAL_CO_sampled points_px.csv')
kf2 = pd.read_csv('TAL_CO_MRE_px.csv')
kf6 = pd.read_csv('TAL_CO_PAC_px.csv')

# P-X-T
# AL Data CH4
df3 = pd.read_csv('AL_ch_sampled points_pxt.csv')
df4 = pd.read_csv('AL_CH_MRE_pxt.csv')
df5 = pd.read_csv('AL_CH_PAC_pxt.csv')

# AL Data CO2
gf3 = pd.read_csv('AL_co_sampled points_pxt.csv')
gf4 = pd.read_csv('AL_CO_MRE_pxt.csv')
gf5 = pd.read_csv('AL_CO_PAC_pxt.csv')

# TAL Data CH4
hf3 = pd.read_csv('TAL_CH_sampled points_pxt.csv')
hf4 = pd.read_csv('TAL_CH_MRE_pxt.csv')
hf5 = pd.read_csv('TAL_CH_PAC_pxt.csv')

# TAL Data CO2
kf3 = pd.read_csv('TAL_CO_sampled points_pxt.csv')
kf4 = pd.read_csv('TAL_CO_MRE_pxt.csv')
kf5 = pd.read_csv('TAL_CO_PAC_pxt.csv')

#========= Define Parameters and Create Plots =========
# P-X Sampled Points
panel_labels = ["(a)", "(b)"]
fig, ax = plt.subplots(1, 2, figsize = (11, 6))

# CH4
ax[0].scatter(hf.loc[21:, 'Mole Frac'], hf.loc[21:, 'Pressure'] / 1e5, marker = '*', s = 50, color = 'steelblue', label = "TAL_Points")
ax[0].scatter(df.loc[21:, 'Mole Frac'], df.loc[21:, 'Pressure'] /1e5, marker = '.', s = 50, color = 'crimson', label = "AL_Points")
ax[0].set_xlabel('Mole Fractions', fontsize = 14)
ax[0].set_ylabel('Pressure (bar)', fontsize = 14)
ax[0].set_title('$CH_{4}$ AL and TAL Sampled Points Comparison', fontsize = 14)
ax[0].tick_params(axis = 'both', labelsize = 12)
ax[0].text(-0.10,1.05,panel_labels[0],transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

# CO2
ax[1].scatter(kf.loc[21:, 'Mole Frac'], kf.loc[21:, 'Pressure'] / 1e5, marker = '*',  s = 50, color = 'steelblue', label = "TAL_Points")
ax[1].scatter(gf.loc[21:, 'Mole Frac'], gf.loc[21:, 'Pressure'] /1e5, marker = '.',  s = 50, color = 'crimson', label = "AL_Points")
ax[1].set_xlabel('Mole Fractions', fontsize = 14)
ax[1].set_ylabel('Pressure (bar)', fontsize = 14)
ax[1].set_title('$CO_{2}$ AL and TAL Sampled Points Comparison', fontsize = 14)
ax[1].tick_params(axis = 'both', labelsize = 12)
ax[1].text(-0.10,1.05,panel_labels[1],transform=ax[1].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

handles, labels = ax.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.98))
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('PX_sampled_points.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('PX_sampled_points.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()

# P-X MRE
fig, ax = plt.subplots(1, 2, figsize = (11, 6))

# CH4
ax[0].plot(hf2.loc[:, 'GP MRE'], marker = '.', markersize = '10', markeredgecolor = 'lightsteelblue', color = 'steelblue', label = "TAL")
ax[0].plot(df2.loc[:, 'GP MRE'], marker = '.', markersize = '10', markeredgecolor = 'lightcoral', color = 'crimson', label = "AL")
ax[0].set_xlabel('Iterations', fontsize = 14)
ax[0].set_ylabel('MRE', fontsize = 14)
ax[0].set_title('$CH_{4}$ AL and TAL GP Predicted MRE Comparison', fontsize = 14)
ax[0].grid(True, color= 'lightgray', linestyle= '--', alpha = 0.4)
ax[0].tick_params(axis = 'both', labelsize = 12)
ax[0].text(-0.10,1.05,panel_labels[0],transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

# CO2
ax[1].plot(kf2.loc[:, 'GP MRE'], marker = '.', markersize = '10', markeredgecolor = 'lightsteelblue', color = 'steelblue', label = "TAL")
ax[1].plot(gf2.loc[:, 'GP MRE'], marker = '.', markersize = '10', markeredgecolor = 'lightcoral', color = 'crimson', label = "AL")
ax[1].set_xlabel('Iterations', fontsize = 14)
ax[1].set_ylabel('MRE', fontsize = 14)
ax[1].set_title('$CO_{2}$ AL and TAL GP Predicted MRE Comparison', fontsize = 14)
ax[1].grid(True, color= 'lightgray', linestyle= '--', alpha = 0.4)
ax[1].tick_params(axis = 'both', labelsize = 12)
ax[1].text(-0.10,1.05,panel_labels[1],transform=ax[1].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

handles, labels = ax.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.98))
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('PX_MRE.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('PX_MRE.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()

# P-X PAC
fig, ax = plt.subplots(1, 2, figsize = (11, 6))

# CH4
ax[0].plot(hf6.loc[:, 'PAC'], marker = '.', markersize = '10', markeredgecolor = 'lightsteelblue', color = 'steelblue', label = "TAL")
ax[0].plot(df6.loc[:, 'PAC'], marker = '.', markersize = '10', markeredgecolor = 'lightcoral', color = 'crimson', label = "AL")
ax[0].set_xlabel('Iterations', fontsize = 14)
ax[0].set_ylabel('PAC', fontsize = 14)
ax[0].set_title('$CH_{4}$ AL and TAL GP Perceived Accuracy Comparison', fontsize = 14)
ax[0].grid(True, color= 'lightgray', linestyle= '--', alpha = 0.4)
ax[0].tick_params(axis = 'both', labelsize = 12)
ax[0].text(-0.10,1.05,panel_labels[0],transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

# CO2
ax[1].plot(kf6.loc[:, 'PAC'], marker = '.', markersize = '10', markeredgecolor = 'lightsteelblue', color = 'steelblue', label = "TAL")
ax[1].plot(gf6.loc[:, 'PAC'], marker = '.', markersize = '10', markeredgecolor = 'lightcoral', color = 'crimson', label = "AL")
ax[1].set_xlabel('Iterations', fontsize = 14)
ax[1].set_ylabel('PAC', fontsize = 14)
ax[1].set_title('$CO_{2}$ AL and TAL GP Perceived Accuracy Comparison', fontsize = 14)
ax[1].grid(True, color= 'lightgray', linestyle= '--', alpha = 0.4)
ax[1].tick_params(axis = 'both', labelsize = 12)
ax[1].text(-0.10,1.05,panel_labels[1],transform=ax[1].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

handles, labels = ax.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.98))
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('PX_PAC.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('PX_PAC.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()

# P-X-T Sampled Points
panel_labels = ["(a)", "(b)"]
fig, ax = plt.subplots(1, 2, figsize = (11, 6))

# CH4
ax[0].scatter(hf3.loc[6:, 'Mole Frac'], hf3.loc[6:, 'Pressure'] / 1e5, marker = '*',  s = 50, color = 'steelblue', label = "TAL_Points")
ax[0].scatter(df3.loc[6:, 'Mole Frac'], df3.loc[6:, 'Pressure'] /1e5, marker = '.',  s = 50, color = 'crimson', label = "AL_Points")
ax[0].set_xlabel('Mole Fractions', fontsize = 14)
ax[0].set_ylabel('Pressure (bar)', fontsize = 14)
ax[0].set_title('$CH_{4}$ AL and TAL Sampled Points Comparison', fontsize = 14)
ax[0].tick_params(axis = 'both', labelsize = 12)
ax[0].text(-0.10,1.05,panel_labels[0],transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

# CO2
ax[1].scatter(kf3.loc[6:, 'Mole Frac'], kf3.loc[6:, 'Pressure'] / 1e5, marker = '*', s = 50, color = 'steelblue', label = "TAL_Points")
ax[1].scatter(gf3.loc[6:, 'Mole Frac'], gf3.loc[6:, 'Pressure'] /1e5, marker = '.', s = 50, color = 'crimson', label = "AL_Points")
ax[1].set_xlabel('Mole Fractions', fontsize = 14)
ax[1].set_ylabel('Pressure (bar)', fontsize = 14)
ax[1].set_title('$CO_{2}$ AL and TAL Sampled Points Comparison', fontsize = 14)
ax[1].tick_params(axis = 'both', labelsize = 12)
ax[1].text(-0.10,1.05,panel_labels[1],transform=ax[1].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

handles, labels = ax.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.98))
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('PXT_sampled_points.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('PXT_sampled_points.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()

# P-X MRE
fig, ax = plt.subplots(1, 2, figsize = (11, 6))

# CH4
ax[0].plot(hf4.loc[:, 'GP MRE'], marker = '.', markersize = '10', markeredgecolor = 'lightsteelblue', color = 'steelblue', label = "TAL")
ax[0].plot(df4.loc[:, 'GP MRE'], marker = '.', markersize = '10', markeredgecolor = 'lightcoral', color = 'crimson', label = "AL")
ax[0].set_xlabel('Iterations', fontsize = 14)
ax[0].set_ylabel('MRE', fontsize = 14)
ax[0].set_title('$CH_{4}$ AL and TAL GP Predicted MRE Comparison', fontsize = 14)
ax[0].grid(True, color= 'lightgray', linestyle= '--', alpha = 0.4)
ax[0].tick_params(axis = 'both', labelsize = 12)
ax[0].text(-0.10,1.05,panel_labels[0],transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

# CO2
ax[1].plot(kf4.loc[:, 'GP MRE'], marker = '.', markersize = '10', markeredgecolor = 'lightsteelblue', color = 'steelblue', label = "TAL")
ax[1].plot(gf4.loc[:, 'GP MRE'], marker = '.', markersize = '10', markeredgecolor = 'lightcoral', color = 'crimson', label = "AL")
ax[1].set_xlabel('Iterations', fontsize = 14)
ax[1].set_ylabel('MRE', fontsize = 14)
ax[1].set_title('$CO_{2}$ AL and TAL GP Predicted MRE Comparison', fontsize = 14)
ax[1].grid(True, color= 'lightgray', linestyle= '--', alpha = 0.4)
ax[1].tick_params(axis = 'both', labelsize = 12)
ax[1].text(-0.10,1.05,panel_labels[1],transform=ax[1].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

handles, labels = ax.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.98))
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('PXT_MRE.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('PXT_MRE.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()

# P-X PAC
fig, ax = plt.subplots(1, 2, figsize = (11, 6))

# CH4
ax[0].plot(hf5.loc[:, 'PAC'], marker = '.', markersize = '10', markeredgecolor = 'lightsteelblue', color = 'steelblue', label = "TAL")
ax[0].plot(df5.loc[:, 'PAC'], marker = '.', markersize = '10', markeredgecolor = 'lightcoral', color = 'crimson', label = "AL")
ax[0].set_xlabel('Iterations', fontsize = 14)
ax[0].set_ylabel('PAC', fontsize = 14)
ax[0].set_title('$CH_{4}$ AL and TAL GP Perceived Accuracy Comparison', fontsize = 14)
ax[0].grid(True, color= 'lightgray', linestyle= '--', alpha = 0.4)
ax[0].tick_params(axis = 'both', labelsize = 12)
ax[0].text(-0.10,1.05,panel_labels[0],transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

# CO2
ax[1].plot(kf5.loc[:, 'PAC'], marker = '.', markersize = '10', markeredgecolor = 'lightsteelblue', color = 'steelblue', label = "TAL")
ax[1].plot(gf5.loc[:, 'PAC'], marker = '.', markersize = '10', markeredgecolor = 'lightcoral', color = 'crimson', label = "AL")
ax[1].set_xlabel('Iterations', fontsize = 14)
ax[1].set_ylabel('PAC', fontsize = 14)
ax[1].set_title('$CO_{2}$ AL and TAL GP Perceived Accuracy Comparison', fontsize = 14)
ax[1].grid(True, color= 'lightgray', linestyle= '--', alpha = 0.4)
ax[1].tick_params(axis = 'both', labelsize = 12)
ax[1].text(-0.10,1.05,panel_labels[1],transform=ax[1].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

handles, labels = ax.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=13, frameon=False, bbox_to_anchor=(0.5, 0.98))
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('PXT_PAC.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('PXT_PAC.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()
