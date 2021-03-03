import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import signal
from scipy import stats
if __name__ == '__main__':
    plt.close('all')
    df = pd.read_csv(r'C:/Users/Jonas/Downloads/OxCGRT_latest.csv', low_memory=False)
    df.loc[:,'Date'] = pd.to_datetime(df['Date'], format='%Y%m%d').values
    df_NZ = df[df['CountryName'] == 'New Zealand'].interpolate(method='nearest').interpolate(method='pad')
    df_US = df[df['CountryName'] == 'United States']

    end_sim_date = '2020.05.27'

    # dates = pd.to_datetime(df['data'])
    # start_offset_fit = np.where(dates.dt.strftime('%Y.%m.%d') == start_fit_date)[0][0]

    df_US = df_US[df_US['Jurisdiction'] == 'NAT_TOTAL'].interpolate(method='nearest').interpolate(method='pad')
    df_NOR = df[df['CountryName'] == 'Norway'].interpolate(method='nearest').interpolate(method='pad')
    df_BR = df[df['CountryName'] == 'Brazil']
    df_BR = df_BR[df_BR['Jurisdiction'] == 'NAT_TOTAL'].interpolate(method='nearest').interpolate(method='pad')
    dfs = [df_NZ, df_US, df_NOR, df_BR]
    legends = ['New Zealand', 'United States', 'Norway', 'Brazil']


    diff_US = np.diff(df_US['ConfirmedCases'][:-1].replace(np.nan, 0))
    diff_NOR = np.diff(df_NOR['ConfirmedCases'][:-1].replace(np.nan, 0))
    diff_BR = np.diff(df_BR['ConfirmedCases'][:-1].replace(np.nan,0))
    diff_NZ = np.diff(df_NZ['ConfirmedCases'][:-1].replace(np.nan, 0))
    diffs = np.array([diff_NZ, diff_US, diff_NOR, diff_BR])


    fs = 1 / 24 / 3600
    nyquist = fs / 2  # 0.5 times the sampling frequency
    cutoff = .01  # fraction of nyquist frequency, here  it is 5 days
    print('cutoff= ', 1 / cutoff * nyquist * 24 * 3600, ' days')  # cutoff=  4.999999999999999  days
    b, a = signal.butter(5, cutoff, btype='lowpass')  # low pass filter

    for k, diff in enumerate(diffs):
        initial = True
        for i, d in enumerate(diff):
            if d != 0:
                initial = False
            if (d == 0) and not initial:
                diff[i] = diff[i-1]
        diffs[k] = diff


    diffs_filt = [np.array(signal.filtfilt(b, a, diff)) for diff in diffs]

    fig, ax = plt.subplots(3)

    diffs[np.where(np.isnan(diffs)[0])] = 0
    [ax[0].plot(df['Date'], df['StringencyIndexForDisplay'], label=leg) for df, leg in zip(dfs, legends)]
    [ax[1].plot(df['Date'][1:-1], diff) for df, diff in zip(dfs, diffs)]
    ax[1].set_yscale('log')
    [ax[2].plot(df['Date'], df['ConfirmedCases'], label=legends[i]) for i, df in enumerate(dfs)]
    ax[2].set_yscale('log')
    ax[2].legend(loc='lower right')
    fig.subplots_adjust(hspace=0.3)
    _ = [x.set_xticklabels([]) for x in ax[:-1]]
    _ = [x.set_title(t) for t, x in zip(['Stringency Indexes', 'New Cases', 'Cumulative Cases'], ax)]
    _ = [x.grid() for x in ax]

    plt.show()

    fig1, ax1 = plt.subplots(4)
    ax1_twin = [x.twinx() for x in ax1]
    # fig1.subplots_adjust(hspace=0.5)
    [x.set_xticklabels('') for x in np.concatenate([ax1[:-1], ax1_twin[:-1]])]
    [ax1[i].plot(df['Date'], df['StringencyIndexForDisplay'], label = leg, color='r') for i, (df, leg) in enumerate(zip(dfs, legends))]
    [ax1_twin[i].plot(df['Date'], df['ConfirmedCases'], label = leg) for i, (df, leg) in enumerate(zip(dfs, legends))]
    # [x.set_ylim([0,100]) for x in ax1_twin]
    [x.grid() for x in ax1]

    ax1[0].legend(legends)

    save = False
    if save:
        fig.savefig('../Data/Stringency_Infected_Comparison.eps', format='eps')

    plt.show()


