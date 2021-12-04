import scipy.stats as st
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
import seaborn as sns
import pandas as pd

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
    print("Ljung-Box test of residuals:")
    print(sm.stats.acorr_ljungbox(ts, lags=[lags], return_df=True))

def boxcox_lambda_plot(ts, window_width):
    # Box-Cox Transformation
    fm = min(ts.values)
    if fm < 0:
        ts = pd.DataFrame(ts.values - min(ts.values) + 1, columns=ts.columns.values.tolist())
    dfm = pd.concat([ts.rolling(window_width).mean(), ts.rolling(window_width).std()], axis=1)
    dfm.columns = ['log_ma','log_sd']
    dfm = dfm.iloc[window_width+1:,:]
    dfm = dfm.reset_index()
    slope, _, r_value, _, _ = st.linregress(dfm['log_ma'], dfm['log_sd'])
    lambd = 1 - slope
    sns.jointplot(x='log_ma', y='log_sd', data=dfm, kind='reg', 
                  marginal_kws=dict(bins=15, rug=True),
                  annot_kws=dict(stat="r"),
                  joint_kws={'line_kws':{'color':'cyan'}, 'scatter_kws':{'alpha': 0.5, 'edgecolor':'black'}})
    plt.gca().text(-0.5, 0, 'Lambda = ' + str(round(lambd,2)) + '. R squared = ' + str(round(r_value,2)) + ', window width = ' + str(round(window_width,2)))
    plt.show()

def inverse_box_cox(time_series, lmbda):
    """
    Inverse Box-Cox Transformation
    """
    if lmbda == 0:
      return(np.exp(time_series))
    else:
      return(np.exp(np.log(lmbda*time_series+1)/lmbda))