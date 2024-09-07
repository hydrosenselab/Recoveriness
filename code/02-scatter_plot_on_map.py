"""
Figure 4 - distribution on map
"""

import geopandas
import pandas as pd
from statsmodels.distributions.empirical_distribution import ECDF
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.colors as mcolors
import matplotlib


custom_gray = matplotlib.colormaps.get_cmap('turbo')
last_100_colors = custom_gray(np.linspace(0.1, 1, 180))  # Adjust the linspace values as needed
custom_gray = mcolors.ListedColormap(last_100_colors)



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
Mean Annual Precipitation
data['precip']
"""
try:
    # Data extraction
    precip_data = (data['precip'])
    # precip_data = np.log10(precip_data)

    mindata = np.min(precip_data)
    maxdata = np.max(precip_data)
    middata = 0.5*(maxdata - mindata)
    lomiddata = 0.5*(middata - mindata) + mindata
    himiddata = 0.5*(maxdata - middata) + middata
    tickdata = [mindata,
                lomiddata,
                middata,
                himiddata,
                maxdata]
    tickdata_str = ["{:.2f}".format(x) for x in tickdata]

    fig, ax = plt.subplots(figsize=(9, 5))
    states.boundary.plot(ax=ax, edgecolor='black')
    points = ax.scatter(data.lon, data.lat,
                        c=precip_data, cmap=custom_gray,
                        marker=".",
                        lw=1,
                        s=15, alpha=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(ls="--")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=-0.5)
    cbar = fig.colorbar(points, cax=cax)
    cbar.set_ticks(tickdata)
    cbar.set_ticklabels(tickdata_str)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Mean Annual Precipitation (mm)', rotation=270)
    # ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("usa_mean_annual_precipitation_.png", dpi=600)
except:
    print("Done!")



"""
Curve Number (Basin)
data['cnbasin']
"""
try:
    # Data extraction
    curve_no_data = (data['cnbasin'])
    # curve_no_data = np.log10(curve_no_data)

    mindata = np.min(curve_no_data)
    maxdata = np.max(curve_no_data)
    middata = 0.5*(maxdata - mindata)
    lomiddata = 0.5*(middata - mindata) + mindata
    himiddata = 0.5*(maxdata - middata) + middata
    tickdata = [mindata,
                lomiddata,
                middata,
                himiddata,
                maxdata]
    tickdata_str = ["{:.2f}".format(x) for x in tickdata]

    fig, ax = plt.subplots(figsize=(9, 5))
    states.boundary.plot(ax=ax, edgecolor='black')
    points = ax.scatter(data.lon, data.lat,
                        c=curve_no_data, cmap=custom_gray,
                        marker=".",
                        lw=1,
                        s=15, alpha=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(ls="--")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=-0.5)
    cbar = fig.colorbar(points, cax=cax)
    cbar.set_ticks(tickdata)
    cbar.set_ticklabels(tickdata_str)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Curve Number (Basin)', rotation=270)
    # ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("usa_curve_number_.png", dpi=600)
except:
    print("Done!")



"""
Slope Index
data['si']
"""
try:
    # Data extraction
    slope_index_data = (data['si'])
    # slope_index_data = np.log10(slope_index_data)
    mindata = np.min(slope_index_data)
    maxdata = np.max(slope_index_data)
    middata = 0.5*(maxdata - mindata)
    lomiddata = 0.5*(middata - mindata) + mindata
    himiddata = 0.5*(maxdata - middata) + middata
    tickdata = [mindata,
                lomiddata,
                middata,
                himiddata,
                maxdata]
    tickdata_str = ["{:.2f}".format(x) for x in tickdata]

    fig, ax = plt.subplots(figsize=(9, 5))
    states.boundary.plot(ax=ax, edgecolor='black')
    points = ax.scatter(data.lon, data.lat,
                        c=slope_index_data, cmap=custom_gray,
                        marker=".",
                        lw=1,
                        s=15, alpha=1)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(ls="--")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=-0.5)
    cbar = fig.colorbar(points, cax=cax)
    cbar.set_ticks(tickdata)
    cbar.set_ticklabels(tickdata_str)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('Slope Index (mm)', rotation=270)
    # ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("usa_slope_index_.png", dpi=600)
except:
    print("Done!")



"""
Basin Area
data['carea']
"""
try:
    # Data extraction
    area_data = data['area']
    # area_data = area_data - np.min(area_data) + 10.1
    area_data = np.log10(area_data)


    mindata = np.min(area_data)
    maxdata = np.max(area_data)
    middata = 0.5*(maxdata - mindata)
    lomiddata = 0.5*(middata - mindata) + mindata
    himiddata = 0.5*(maxdata - middata) + middata
    tickdata = [mindata,
                lomiddata,
                middata,
                himiddata,
                maxdata]
    tickdata_str = ["{:.2f}".format(x) for x in tickdata]

    fig, ax = plt.subplots(figsize=(9, 5))
    states.boundary.plot(ax=ax, edgecolor='black')
    points = ax.scatter(data.lon, data.lat,
                        c=area_data, cmap=custom_gray,
                        marker=".",
                        lw=1,
                        s=15, alpha=1,
                        norm="log")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(ls="--")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=-0.5)
    cbar = fig.colorbar(points, cax=cax)
    cbar.set_ticks(tickdata)
    cbar.set_ticklabels(tickdata_str)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('log$_{' + str(10) + '}$ [(Basin Area (km$^2$)]', rotation=270)
    # ax.axis('off')
    plt.tight_layout()
    plt.show()
    plt.savefig("usa_basin_area_.png", dpi=600)
except:
    print("Done!")