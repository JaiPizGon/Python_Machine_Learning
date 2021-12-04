import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
plt.style.use('ggplot')
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.pipeline import Pipeline

def PlotDataframe(df, output_name, factor_levels=7, figsize=(19,15), bins=30, spline_degree=7):    
    def make_polynomial_regression(degree):
        return Pipeline([
            ('poly', Polynomial(degree=degree)),
            ('regression', LinearRegression(fit_intercept=True))
        ])
    nplots = df.shape[1]
    out_num = np.where(df.columns.values == output_name)[0]
    out_factor = False
    # Check if the output is categorical
    if df[output_name].dtype.name == 'category' or len(df[output_name].unique()) <= factor_levels:
        out_factor = True
        df[output_name] = df[output_name].astype('category')

    # Create subplots
    fig, axs = plt.subplots(math.floor(math.sqrt(nplots))+1, math.ceil(math.sqrt(nplots)), figsize=figsize)
    fig.tight_layout(pad=4.0)
    
    if out_factor:
        input_num = 0
        for ax in axs.ravel():
            if input_num < nplots:
                # Create plots
                if input_num == out_num:
                    df.groupby(output_name).size().plot.bar(ax=ax, rot=0)
                    ax.set_title('Histogram of ' + output_name)
                else:
                    if df.iloc[:,input_num].dtype.name == 'category':
                        df.groupby([output_name,df.columns.values.tolist()[input_num]]).size().unstack().plot(kind='bar', ax=ax, rot=0)
                        ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)
                    elif df.iloc[:,input_num].dtype.name in ['int64', 'float64']:
                        df.pivot(columns=output_name, values=df.columns.values.tolist()[input_num]).plot.hist(bins=bins, ax=ax,rot=0)
                        ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)

                input_num += 1
            else:
                ax.axis('off')

    else:
        input_num = 0
        for ax in axs.ravel():
            if input_num < nplots:
                # Create plots
                if input_num == out_num:
                    df[output_name].plot.hist(bins=bins,ax=ax)
                    ax.set_title('Histogram of ' + output_name)
                else:
                    if df.iloc[:,input_num].dtype.name == 'category':
                        sns.boxplot(x=df.columns.values.tolist()[input_num], y=output_name, data=df, ax=ax)
                        ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)
                    elif df.iloc[:,input_num].dtype.name in ['int64', 'float64']:
                        # Train a polynomial model of the degree chosen
                        sns.regplot(x=df.columns.values.tolist()[input_num], y=output_name, data=df, order=spline_degree, scatter_kws={'alpha': 0.5, 'color':'black'},line_kws={'color':'navy'}, ax=ax)
                        ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)
                    else:
                        sns.scatterplot(x=df.columns.values.tolist()[input_num], y=output_name, data=df, alpha=0.5, color='black', ax=ax)
                        ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)
                input_num += 1
            else:
                ax.axis('off')

    # Plot the plots created
    plt.show()

# Summary for Linear Regression model from scikit
def summaryLinReg(model, X_train, y_train):
    # Obtain coefficients of the model
    if type(model) is LinearRegression:
        coefs = model.coef_
        intercept = model.intercept_
    else:
        coefs = model[len(model) - 1].coef_ #We suppose last position of pipeline is the linear regression model
        intercept = model[len(model) - 1].intercept_

    coefs = np.append(intercept,coefs)
    # Obtain names of the inputs
    if X_train.select_dtypes('category').shape[1] > 0:
        input_names = []
        for cat_input in X_train.select_dtypes('category').columns:
            input_names += [cat_input + str(cat) for cat in X_train[cat_input].unique()]
        coefnames = X_train.select_dtypes(exclude='category').columns.values.tolist() + input_names
    else:
        coefnames = X_train.columns.values.tolist()
    coefnames.insert(0,'Intercept')
    # Calculate matrix of predicted class probabilities.
    # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
    y_pred = model.predict(X_train)
    res = y_train - y_pred
    print('Residuals:')
    quantiles = np.quantile(res, [0,0.25,0.5,0.75,1], axis=0)
    quantiles = pd.DataFrame(quantiles, index=['Min','1Q','Median','3Q','Max'])
    print(quantiles.transpose())
    # Print coefficients of the model
    print('\nCoefficients:')
    coefs = pd.DataFrame(data=coefs, index=coefnames, columns=['Estimate'])

    if type(model) is Pipeline:
        X_train = model[0].transform(X_train)

    ## Calculate std error of inputs ------------- 
    if type(X_train).__module__ == np.__name__:
        X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    else:
        X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train.toarray()])
    X_design = pd.DataFrame(X_design, columns=coefnames)

    ols = sm.OLS(y_train.values, X_design)
    ols_result = ols.fit()
    print(ols_result.summary())
    return

def plotModelDiagnosis(df, pred, output_name, figsize=(19,15), bins=30, spline_degree=5):
    # Create the residuals
    df['residuals'] = df[output_name] - df[pred]
    out_num = np.where(df.columns.values == 'residuals')[0]
    nplots = df.shape[1]
    # Create subplots
    fig, axs = plt.subplots(math.floor(math.sqrt(nplots))+1, math.ceil(math.sqrt(nplots)), figsize=figsize)
    fig.tight_layout(pad=4.0)

    input_num = 0
    for ax in axs.ravel():
        if input_num < nplots:
            # Create plots
            if input_num == out_num:
                df['residuals'].plot.hist(bins=bins, ax=ax)
                ax.set_title('Histogram of residuals')
            else:
                if df.iloc[:,input_num].dtype.name == 'category':
                    sns.boxplot(x=df.columns.values.tolist()[input_num], y='residuals', data=df, ax=ax)
                    ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + 'residuals')
                elif df.iloc[:,input_num].dtype.name in ['int64', 'float64']:
                    sns.regplot(x=df.columns.values.tolist()[input_num], y='residuals', data=df, ax=ax, order=spline_degree, ci=None, scatter_kws={'alpha': 0.5, 'color':'black'}, line_kws={'color':'navy'})
                    ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + 'residuals')
                else:
                    sns.scatterplot(x=df.columns.values.tolist()[input_num], y='residuals', data=df, color='black', alpha = 0.5, ax=ax)
                    ax.set_title(df.columns.values.tolist()[input_num] + ' vs ' + output_name)

            input_num += 1
        else:
            ax.axis('off')

def dotplot(scores, metric:str):
    plt.xlabel(metric)
    plt.ylabel('')
    plt.title("Scores")
    scores_list = [score for key, score in scores.items()]
    for i in range(len(scores_list)):
        plt.boxplot(scores_list[i], positions=[i], vert=False)
    plt.yticks(list(range(len(scores_list))), list(scores.keys()))
    plt.show()