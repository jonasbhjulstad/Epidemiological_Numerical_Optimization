import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats
if __name__ == '__main__':
    df = pd.read_csv(r'C:/Users/Jonas/Downloads/OxCGRT_latest.csv', low_memory=False)
    df.loc[:,'Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').values
    df_NZ = df[df['CountryName'] == 'New Zealand'].interpolate(method='nearest').interpolate(method='pad')
    df_US = df[df['CountryName'] == 'United States']


    df_US = df_US[df_US['Jurisdiction'] == 'NAT_TOTAL'].interpolate(method='nearest').interpolate(method='pad')
    df_NOR = df[df['CountryName'] == 'Norway'].interpolate(method='nearest').interpolate(method='pad')
    df_BR = df[df['CountryName'] == 'Brazil']
    df_BR = df_BR[df_BR['Jurisdiction'] == 'NAT_TOTAL'].interpolate(method='nearest').interpolate(method='pad')
    dfs = [df_NZ, df_US, df_NOR, df_BR]
    legends = ['New Zealand', 'United States', 'Norway', 'Brazil']

    diff_US = np.diff(df_US['ConfirmedCases'][:-1])
    diff_NOR = np.diff(df_NOR['ConfirmedCases'][:-1])
    diff_BR = np.diff(df_BR['ConfirmedCases'][:-1])
    diff_NZ = np.diff(df_NZ['ConfirmedCases'][:-1])
    diffs = [diff_NZ, diff_US, diff_NOR, diff_BR]


    fig, ax = plt.subplots(3)

    [ax[0].plot(df['Date'], df['StringencyIndexForDisplay'], label=leg) for df, leg in zip(dfs, legends)]
    [ax[1].plot(df['Date'][1:-1], diff) for df, diff in zip(dfs, diffs)]
    ax[1].set_yscale('log')
    [ax[2].plot(df['Date'][1:], np.diff(df['ConfirmedCases'])*df['StringencyIndexForDisplay'][:-1]) for df in dfs]
    ax[0].legend()
    plt.show()



    fig1, ax1 = plt.subplots(4)
    ax1_twin = [x.twinx() for x in ax1]
    [ax1[i].plot(df['Date'], df['StringencyIndexForDisplay'], label = leg) for i, (df, leg) in enumerate(zip(dfs, legends))]
    [ax1_twin[i].plot(df['Date'], df['ConfirmedCases'], label = leg) for i, (df, leg) in enumerate(zip(dfs, legends))]
    # [x.set_ylim([0,100]) for x in ax1_twin]


    plt.show()


