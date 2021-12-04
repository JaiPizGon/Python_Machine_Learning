import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np

def ts_display(ts, lags=25):
    grid = plt.GridSpec(2, 2)
    # Plot series as is
    plt.subplot(grid[0, :])
    plt.plot(ts, color='black')
    # Plot ACF
    ax = plt.subplot(grid[1, 0])
    sm.graphics.tsa.plot_acf(ts.values.squeeze(), lags=lags, ax=ax)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    # Plot PACF
    ax = plt.subplot(grid[1, 1])
    sm.graphics.tsa.plot_pacf(ts.values.squeeze(), lags=lags, ax=ax)
    ax.set_xlabel('Lag')
    ax.set_ylabel('PACF')

def check_residuals(ts, lags=25, bins=30):
    grid = plt.GridSpec(2, 2)
    # Plot series as is
    plt.subplot(grid[0, 0])
    plt.plot(ts, color='black')
    # Plot histogram of residuals
    ax = plt.subplot(grid[0, 1])
    ts.plot(kind='hist', ax=ax, bins=bins, density=True, edgecolor='black', color='gray', label='Data',legend=False)
    ts.plot(kind='kde', ax=ax, label='PDF', color='red',legend=False)
    plt.ylabel('Probability')
    # Plot ACF
    ax = plt.subplot(grid[1, 0])
    sm.graphics.tsa.plot_acf(ts.values.squeeze(), lags=lags, ax=ax)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    # Plot PACF
    ax = plt.subplot(grid[1, 1])
    sm.graphics.tsa.plot_pacf(ts.values.squeeze(), lags=lags, ax=ax)
    ax.set_xlabel('Lag')
    ax.set_ylabel('PACF')