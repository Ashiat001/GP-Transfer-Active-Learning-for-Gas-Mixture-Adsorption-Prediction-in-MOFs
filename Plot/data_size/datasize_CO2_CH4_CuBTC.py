import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#======= Read in Performance Metrics Files =======
# P-X Test
ch_px = pd.read_csv('Metrics_pure_ch_ch_px.csv', header=None)
co_px = pd.read_csv('Metrics_pure_co_co_px.csv', header=None)
# P-X-T Test
ch_pxt = pd.read_csv('Metrics_pure_ch_ch_pxt.csv', header= None)
co_pxt = pd.read_csv('Metrics_pure_co_co_pxt.csv', header= None)
# print(co_pxt)

#======= Define Plot Values and Generate Plots ========
# P-X Values
data_size_px_ch = ch_px.iloc[:, 0]
data_size_px_co = co_px.iloc[:, 0]

# CO2 metrics
mre_co2_px = co_px.iloc[:, 3]
no_iters_co2_px = co_px.iloc[:, 4]

# CH4 metrics
mre_ch4_px = ch_px.iloc[:, 3]
no_iters_ch4_px = ch_px.iloc[:, 4]

# P-X-T Values
data_size_pxt_ch = ch_pxt.iloc[:, 0]
data_size_pxt_co = co_pxt.iloc[:, 0]

# CO2 metrics
mre_co2_pxt = co_pxt.iloc[:, 3]
no_iters_co2_pxt = co_pxt.iloc[:, 4]

# CH4 metrics
mre_ch4_pxt = ch_pxt.iloc[:, 3]
no_iters_ch4_pxt = ch_pxt.iloc[:, 4]

# Generate Plot
panel_labels = ["(a)", "(b)"]
fig, ax = plt.subplots(figsize=(15, 6), ncols=2)

scale = 3  # adjust bubble size

# P-X Plot
for i, ds in enumerate(data_size_px_co):
    ax[0].scatter(ds, mre_co2_px[i] *100, s=no_iters_co2_px[i] * scale, 
               color='steelblue', alpha=0.5, zorder=5, edgecolors = 'darkblue')
for i, ds in enumerate(data_size_px_ch):
    ax[0].scatter(ds, mre_ch4_px[i]*100, s=no_iters_ch4_px[i] * scale, 
               color='darkgreen', alpha=0.5, zorder=5, edgecolors = 'black')

# Connect bubbles with a thin line to show trend
ax[0].plot(data_size_px_co, mre_co2_px*100, color='steelblue', linestyle='--', linewidth=2.0,label = r'$\mathrm{CO}_{2}$')
ax[0].plot(data_size_px_ch, mre_ch4_px*100, color='darkgreen', linestyle='--', linewidth=2.0, label= r'$\mathrm{CH}_{4}$')

# Size legend
for iters in [min(no_iters_co2_px), max(no_iters_co2_px)]:
    ax[0].scatter([], [], s=iters * scale, color='grey', alpha=0.5,
               label=f'{iters} TAL iters')

ax[0].set_xlabel('Initial Training Data Size', fontsize = 15, fontweight="bold")

for tick in ax[0].get_xticklabels() + ax[0].get_yticklabels():
    tick.set_fontsize(13)
    tick.set_fontweight("bold")

ax[0].set_ylabel('MAPE (%)', fontsize = 15, fontweight="bold")
# ax[0].set_title('TAL Performance vs Starting Mixture Data Size - P-X Space', fontsize = 15)
ax[0].legend(loc='upper right', fontsize = 12)
ax[0].grid(True, linestyle='--', alpha=0.4)
ax[0].text(-0.10,1.02,panel_labels[0],transform=ax[0].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)

# P-X-T Plot
for i, ds in enumerate(data_size_pxt_co):
    ax[1].scatter(ds, mre_co2_pxt[i]*100, s=no_iters_co2_pxt[i] * scale, 
               color='steelblue', alpha=0.5, zorder=5, edgecolors = 'darkblue')
for i, ds in enumerate(data_size_pxt_ch):
    ax[1].scatter(ds, mre_ch4_pxt[i]*100, s=no_iters_ch4_pxt[i] * scale, 
               color='darkgreen', alpha=0.5, zorder=5, edgecolors = 'black')

# Connect bubbles with a thin line to show trend
ax[1].plot(data_size_pxt_co, mre_co2_pxt*100, color='steelblue',  linestyle='--', linewidth=2.0, label=r'$\mathrm{CO}_{2}$')
ax[1].plot(data_size_pxt_ch, mre_ch4_pxt*100, color='darkgreen', linestyle='--', linewidth=2.0, label=r'$\mathrm{CH}_{4}$')

# Size legend
for iters in [min(no_iters_co2_pxt), max(no_iters_co2_pxt)]:
    ax[1].scatter([], [], s=iters * scale, color='grey', alpha=0.5,
               label=f'{iters} TAL iters')

ax[1].set_xlabel('Initial Training Data Size', fontsize = 15, fontweight="bold")
ax[1].set_ylabel('MAPE (%)', fontsize = 15, fontweight="bold")

for tick in ax[1].get_xticklabels() + ax[1].get_yticklabels():
    tick.set_fontsize(13)
    tick.set_fontweight("bold")

# ax[1].set_title('TAL Performance vs Starting Mixture Data Size - P-X-T Space', fontsize = 15)
ax[1].legend(loc='best', fontsize = 12)
ax[1].grid(True, linestyle='--', alpha=0.4)
ax[1].text(-0.115,1.02,panel_labels[1],transform=ax[1].transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        ha="left",
        clip_on=False)
plt.tight_layout()
plt.savefig('Data_size_test.png', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.savefig('Data_size_test.pdf', dpi=600, bbox_inches="tight", pad_inches=0.05, facecolor="white")
plt.show()



