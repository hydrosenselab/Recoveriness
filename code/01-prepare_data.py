import geopandas
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns

states = geopandas.read_file('../map_data/usa-states-census-2014.shp')
print(states.head())
print(states.crs)

# states = states.to_crs("EPSG:3395")
soa_shape_map_geo = states.to_crs(epsg=4326)  # EPSG 4326 = WGS84 = https://epsg.io/4326
print(states.iloc[0].geometry.centroid.y, soa_shape_map_geo.iloc[0].geometry.centroid.y)



# Font setting parameters
font = {'size': 14, 'family': 'serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 9}
plt.rcParams.update(params)
plt.rc('text', usetex=False)

data = pd.read_csv("../data/allevents.csv")
"""
# fips:                                     ID
# gauge:                                    ID
# start:                                    Event start time
# peakq:                                    Peak Flow
# peakt:                                    Time of peak flow
# dt:                                       Time difference start of the event and peak flow
# area/carea:                               Corrected (using basin delineation?) basin area
# q2/5/..N..500 :                           Return Period - Ny
# action/ minor.. Major:                    Flood stage threshold
# alpha, beta, cc :                         Kinematic wave parameters
# usgs_area, error:                         True drainage area 
# el,k,rl,rr,si,rdd,rbm, rfocf:             elongation, shape factor, river length, slope index, drainage density,....
# precip, temp, cnbasin,......bpartexture:  curve no., rock , soil
# dc,ldd,..,ruggedness:                     dia, drainage stats
# fd, rt :                                  flow duration, recession time (peak to end)
# nfd,nrtntp,nq:                            Normalized (Unit) ...
# f:                                        Flashiness
"""

"""
Extract data, compute ECDF and Recoveriness
"""
df = data[['gauge', 'peakq', 'action', 'rt', 'area']].copy()
df['r'] = (data['peakq'] - data['action']) / (data['rt'] * data['area'])
df['mr'] = data['gauge']
df_median = df[['gauge', 'r']]
df_median = df_median.groupby('gauge').median().reset_index()
rename_dict = df_median.set_index('gauge').to_dict()['r']
df['mr'] = df.mr.replace(rename_dict)
ecdf = ECDF(df['mr'])
df['mr.ecdf'] = ecdf(df['mr'])

"""
Replace in original dataframe
"""
data['r'] = df['r']
data['mr'] = df['mr']
data['mr.ecdf'] = df['mr.ecdf']
# data['recoveriness'] = np.where(data['mr.ecdf'] >= 0.75, 1, 0)
data['recoveriness'] = np.where(data['mr.ecdf'] >= 0.75, 'Quick', 'Slow')


# Write the data into hard disk
data2 = data.copy(deep=False)
data2.drop(labels=["r", "mr"], axis=1, inplace=True)
data2.dropna()
data2.to_csv(path_or_buf="allcalval_recoveriness.csv",
             sep=",")

"""
Plot on map
"""
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    states.boundary.plot(ax=ax, edgecolor='black')
    points = ax.scatter(data.lon, data.lat, c=data['mr.ecdf'], cmap="turbo",
                         lw=0,
                         s=15, alpha=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    # plt.grid(ls="--")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=-0.5)
    cbar = fig.colorbar(points, cax=cax)
    cbar.set_ticks([0.00, 0.25, 0.50, 0.75, 1.00])
    cbar.set_ticklabels([0.0, 0.25, 0.50, 0.75, 1.0])
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Recoveriness', rotation=270)
    # ax.axis('off')
    plt.tight_layout()
    plt.show()
    # plt.savefig("usa_recovery_.pdf", dpi=300)
    plt.savefig("usa_recovery_ng_.png", dpi=600)
except:
    print("Done!")


"""
Null operations
"""
data.isnull().sum().sum()
data = data.replace([np.inf, -np.inf], np.nan)
data = data.dropna(axis=0, how='any', subset=None, inplace=False)
data.isnull().sum().sum()
print(data.info())

"""
Extracting relevant columns
"""
data_relevant = data[["gauge", "lat", "lon", "start", "end",
                      "peakq", "peakt", "dt", "area", "carea",
                      "action", "minor", "moderate", "major", "regulation",
                      "alpha", "beta", "cc", "usgs_area", "est_area",
                      "error", "el", "k", "rl", "rr",
                      "si", "rdd", "rbm", "rfocf", "slopeoutlet",
                      "precip", "temp", "cnbasin", "cncell", "coemcell",
                      "imperviousbasin", "imperviouscell", "kfact", "rockdepth", "rockvolume",
                      "bpartexture", "dc", "ldd", "lbm", "lfocf",
                      "ruggedness", "fd", "tp", "rt", "nfd",
                      "ntp", "nrt", "nq", "county", "class",
                      "prop", "state", "month", "year", "season",
                      "maxseason", "r", "mr", "mr.ecdf", "recoveriness"
                      ]]

data_selected = data[[
    "area", "el", "k", "rl", "rr",
    "si", "slopeoutlet", "precip",
    "temp", "kfact", "rockdepth",
    "rockvolume", "bpartexture",
    "cnbasin", "mr.ecdf"]]


"""
Save the dataframe
"""
# data.to_csv(path_or_buf="data_prepared.csv", sep=",")
data_selected.to_csv(path_or_buf="data_prepared.csv", sep=",")

print("Done!")


# """
# Correlation
# """
# corr_matrix = data_selected.corr()
# fig, ax = plt.subplots(figsize=(9, 9))
# sns.heatmap(data=corr_matrix,
#             annot=False, square=True,
#             ax=ax, cbar_kws={'shrink': 0.733})
# plt.tight_layout()
# plt.show()

print("Done!")
