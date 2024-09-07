"""
Quantile plots by windowing
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Font setting parameters
font = {'size': 14, 'family': 'serif', 'serif': ['Palatino']}
plt.rc('font', **font)
params = {'legend.fontsize': 14, 'lines.markersize': 9}
plt.rcParams.update(params)
plt.rc('text', usetex=False)


def quantilePlot(df, xvar, yvar, smoothFactor, varname, xtitle="", ytitle=""):
    # Data Preparation
    df_sub = df[[xvar, yvar]]
    quantiles_x = df_sub[xvar].quantile(np.arange(0, 1, smoothFactor))
    df_quants = []

    # Quantile Calculations
    for i in range(len(quantiles_x) - 1):
        subset = df_sub[(df_sub[xvar] >= quantiles_x.iloc[i]) &
                        (df_sub[xvar] <= quantiles_x.iloc[i + 1])]
        if subset.shape[0] > 100:
            q01 = subset[yvar].quantile(0.01)
            q10 = subset[yvar].quantile(0.10)
            q25 = subset[yvar].quantile(0.25)
            q50 = subset[yvar].quantile(0.50)
            q75 = subset[yvar].quantile(0.75)
            q90 = subset[yvar].quantile(0.90)
            q99 = subset[yvar].quantile(0.99)
            df_quants.append([quantiles_x.iloc[i], q01, q10, q25, q50, q75, q90, q99])

    # Convert to DataFrame and Melt
    df_quants = pd.DataFrame(df_quants, columns=["xvars", "q01", "q10", "q25", "q50", "q75", "q90", "q99"])
    df_melted = df_quants.melt(id_vars="xvars", value_vars=df_quants.columns[1:])

    # Plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_sub, x=xvar, y=yvar, size=0.1, color='b', marker=".", alpha=1)
    # sns.lineplot(data=df_melted, x="xvars", y="value", color='k')
    plt.plot(df_quants["xvars"], df_quants["q50"], "-", color="red")
    plt.fill_between(df_quants["xvars"], df_quants["q01"], df_quants["q99"],
                     color="lightgray", edgecolor='k', alpha=0.3)
    plt.fill_between(df_quants["xvars"], df_quants["q10"], df_quants["q90"],
                     color="darkgray", edgecolor='k', alpha=0.5)
    plt.fill_between(df_quants["xvars"], df_quants["q25"], df_quants["q75"],
                     color="dimgray", edgecolor='k', alpha=0.8)

    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    # plt.xlim([np.min(df_quants["xvars"]), 5000])
    plt.xlim([np.min(df_quants["xvars"]),
              np.max(df_quants["xvars"])])
    plt.grid(ls='--')
    plt.legend('', frameon=False)
    plt.show()
    plt.savefig("qntr-" + varname + ".png", dpi=300)
    print()


# Example Usage
df = pd.read_csv(filepath_or_buffer="data_prepared.csv", sep=",", index_col=0)

xvar, yvar = "area", "mr.ecdf"
quantilePlot(df, xvar, yvar, smoothFactor=0.07, varname=xvar, xtitle="Area (km$^2$)", ytitle="Recoveriness")

xvar, yvar = "precip", "mr.ecdf"
quantilePlot(df, xvar, yvar, smoothFactor=0.07, varname=xvar, xtitle="Precipitation", ytitle="Recoveriness")

xvar, yvar = "si", "mr.ecdf"
quantilePlot(df, xvar, yvar, smoothFactor=0.07, varname=xvar, xtitle="Slope Index", ytitle="Recoveriness")

xvar, yvar = "cnbasin", "mr.ecdf"
quantilePlot(df, xvar, yvar, smoothFactor=0.07, varname=xvar, xtitle="Curve Number (basin)", ytitle="Recoveriness")

print("Done!")
