import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import matplotlib.colors as mcolors
import numpy as np

# Font setting parameters
font = {'size': 14, 'family': 'serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 9}
plt.rcParams.update(params)
plt.rc('text', usetex=False)

def turbo_shifted(shift=0.8):
    colors = []
    num_colors = 256
    midval = 0.5
    original = plt.get_cmap('turbo', num_colors)
    mid_idx = int(midval * len(original.colors))
    ncolors = int(shift * len(original.colors))
    colors_above = original.colors[:mid_idx]
    colors_below = original.colors[mid_idx:]

    colors_above = mcolors.LinearSegmentedColormap.from_list(name="cmap_above",
                                                             colors=list(colors_above),
                                                             N=256 - ncolors)
    colors_below = mcolors.LinearSegmentedColormap.from_list(name="cmap_below",
                                                             colors=list(colors_below),
                                                             N=ncolors)

    merged_colors = colors_above(np.linspace(0, 1, 256))
    merged_colors[:ncolors] = colors_above(np.linspace(0, 1, ncolors))
    merged_colors[ncolors:] = colors_below(np.linspace(0, 1, 256 - ncolors))

    custom_colormap = mcolors.LinearSegmentedColormap.from_list(name="custom",
                                                                colors=merged_colors,
                                                                N=256)
    return custom_colormap

norm = mcolors.TwoSlopeNorm(vmin=0, vmax=1, vcenter=0.5)
custom_turbo = turbo_shifted(shift=0.75)

df_test = pd.read_csv(filepath_or_buffer="df_test.csv")

states = gpd.read_file('../map_data/usa-states-census-2014.shp')
soa_shape_map_geo = states.to_crs(epsg=4326)

fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 10))

recovery_data = (df_test['Recoveriness'])
states.boundary.plot(ax=ax[0], edgecolor='black')
points0 = ax[0].scatter(df_test.lon, df_test.lat,
                        c=recovery_data, cmap=custom_turbo, norm=norm,
                        lw=0, s=1, alpha=1)
ax[0].set_xlabel("Longitude")
ax[0].set_ylabel("Latitude")
ax[0].grid(ls="--")

piw_data = (df_test['piw'])
states.boundary.plot(ax=ax[1], edgecolor='black')
points1 = ax[1].scatter(df_test.lon, df_test.lat,
                        c=piw_data, cmap=custom_turbo, norm=norm,
                        lw=0, s=1, alpha=1)
ax[1].set_xlabel("Longitude")
ax[1].set_ylabel("Latitude")
ax[1].grid(ls="--")

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.12, 0.05, 0.85])
tickdata = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
tickdata_str = ["{:.2f}".format(x) for x in tickdata]

cbar = fig.colorbar(points0, cax=cbar_ax)
cbar.set_ticks(tickdata)
cbar.set_ticklabels(tickdata_str)
cbar.ax.get_yaxis().labelpad = 15

plt.tight_layout()
plt.show()

plt.savefig("usa_uncertainty_recoveriness_same_colorbar.png", dpi=600)

print("Done!")
