# Import necessary modules
import numpy as np
import math
import pandas as pd
from scipy.stats import norm, binom_test
from statsmodels.stats import contingency_tables as cont_tab
import seaborn as sns
import itertools 
from numpy.matlib import repmat
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import confusion_matrix
import sklearn.metrics as mtrs
import matplotlib.collections 
import numbers
import six
from sklearn.calibration import calibration_curve
import matplotlib.lines as mlines
import matplotlib.transforms as mtransforms

# Summary for Logistic Regression model from scikit
def summaryLogReg(model, X_train, y_train):
    # Obtain coefficients of the model
    coefs = model.coef_[0]
    intercept = model.intercept_
    coefs = np.append(intercept,coefs)
    # Obtain names of the inputs
    coefnames = [column for column in X_train.columns]
    coefnames.insert(0,'Intercept')
    # Calculate matrix of predicted class probabilities.
    # Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
    predProbs = model.predict_proba(X_train)
    y_pred = predProbs[:,1]
    y_int = y_train.cat.codes.to_numpy()
    res = y_int - y_pred
    print('Deviance Residuals:')
    quantiles = np.quantile(res, [0,0.25,0.5,0.75,1], axis=0)
    quantiles = pd.DataFrame(quantiles, index=['Min','1Q','Median','3Q','Max'])
    print(quantiles.transpose())
    # Print coefficients of the model
    print('\nCoefficients:')
    coefs = pd.DataFrame(data=coefs, index=coefnames, columns=['Estimate'])
    ## Calculate std error of inputs ------------- 
    X_design = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    # Initiate matrix of 0's, fill diagonal with each predicted observation's variance
    V = np.diagflat(np.product(predProbs, axis=1))
    # Covariance matrix
    covLogit = np.linalg.inv(X_design.T @ V @ X_design)
    # Std errors
    coefs['Std. Err'] = np.sqrt(np.diag(covLogit))
    # t-value
    coefs['t-value'] = coefs['Estimate'] / coefs['Std. Err']
    # P-values
    coefs['Pr(>|t|)'] = (1 - norm.cdf(abs(coefs['t-value']))) * 2
    coefs['Signif'] = coefs['Pr(>|t|)'].apply(lambda x: '***' if x < 0.001 else ('**' if x < 0.01 else ('*' if x < 0.05 else ('.' if x < 0.1 else ' '))))
    print(coefs)
    print('---\nSignif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1')
    ## AIC criterion ----------------
    # Obtain rank of the model
    rank = len(coefs)
    likelihood = y_pred * y_int + (1 - y_pred) * (1 - y_int)
    AIC = 2*rank - 2*math.log(likelihood.max())
    print('AIC:',AIC,' (no es fiable, revisar formula de AIC)')
    return
    
def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

def plot2DClass(X, y, model, var1:str, var2:str, selClass=None, np_grid=200, real=True):
    if X.shape[1] < 2:
        raise ValueError("X must have at least 2 input variables")
    if isinstance(X, pd.DataFrame):
        c_names = [column for column in X.columns]
        if any(not elem in c_names for elem in [var1, var2]):
            nelems = [not elem in c_names for elem in [var1, var2]]
            err_str = " and ".join(list(itertools.compress([var1,var2], nelems))) + ' could not be found in X.'
            raise ValueError(err_str)

    # Predict input data
    df = X.copy()
    df['pred'] = model.predict(X)

    # Check positive class
    if selClass is None:
        selClass = y.unique()[0]
        warnings.warn('The first level of the output would be use as positive class', category=UserWarning)

    if len(df.columns) == 3:
        np_X1 = np.linspace(df[var1].min(), df[var1].max(), np_grid)
        np_X2 = np.linspace(df[var2].min(), df[var2].max(), np_grid)
        X, Y = np.meshgrid(np_X1, np_X2)
        # grid_X1_X2 = pd.DataFrame(CT.expandgrid(np_X1, np_X2))
        grid_X1_X2 = pd.DataFrame(np.c_[X.ravel(), Y.ravel()], columns=[var1,var2])
        # Predict each point of the grid
        grid_X1_X2['pred'] = model.predict(grid_X1_X2)
        grid_X1_X2.columns = [var1, var2, 'Y']
        # Obtain probabilites of the model in the grid and add to the grid data frame
        probabilities = model.predict_proba(grid_X1_X2[[var1,var2]])
        grid_X1_X2 = grid_X1_X2.join(pd.DataFrame(probabilities))
        grid_X1_X2.columns = [var1, var2, 'Y'] + ['_'.join(['prob',lev]) for lev in y.unique()]
        # Define output class variable in grid
        grid_X1_X2['prob'] = grid_X1_X2['_'.join(['prob',selClass])]
        # Classification of input space
        plt.subplot(221)
        sns.scatterplot(x=var1, y=var2, hue='Y', data=grid_X1_X2).set_title('Classification of input space')
        # Probabilities estimated for input space
        plt.subplot(222)  
        y_prob = grid_X1_X2['prob'].to_numpy().reshape(np_grid,np_grid)
        sns.scatterplot(x=var1, y=var2, hue='prob', data=grid_X1_X2).set_title(' '.join(["Probabilities estimated for input space, class:",selClass]))
        
        
        # Classification results
        df2 = pd.concat([df.reset_index(),interaction(y, df['pred']).reset_index()], axis=1)
        del df2['index']
        plt.subplot(223)
        sns.scatterplot(x=var1, y=var2, hue='inter', data= df2).set_title(' '.join(["Classification results, class:", selClass]))
        plt.subplot(224)
        df['Y'] = y
        sns.scatterplot(x=var1, y=var2, hue='Y', data=df).set_title(' '.join(['classes and estimated probability contour lines for class:', selClass]))
        cnt = plt.contour(X, Y, y_prob, colors='black')
        plt.clabel(cnt, inline=True, fontsize=8)
        plt.show()
    return

def interaction(var1, var2, returntype='Series'):
    dVar1 = pd.get_dummies(var1)
    dVar2 = pd.get_dummies(var2)
    names1 = dVar1.columns[np.concatenate(repmat(np.arange(len(dVar1.columns)), len(dVar2.columns), 1).transpose())]
    names2 = dVar2.columns[np.concatenate(repmat(np.arange(len(dVar2.columns)), 1, len(dVar1.columns)).transpose())]
    namesdef = ["_".join([name1,name2]) for name1, name2 in zip(names1, names2)]
    inter = pd.DataFrame(np.multiply(dVar1.iloc[:, np.concatenate(repmat(np.arange(len(dVar1.columns)), len(dVar2.columns), 1).transpose())].to_numpy(), dVar2.iloc[:, np.concatenate(repmat(np.arange(len(dVar2.columns)), 1, len(dVar1.columns)).transpose())].to_numpy()), columns = namesdef)
    if returntype == 'Series':
        return pd.Series(inter.idxmax(axis=1), name='inter')
    else:
        return inter

def confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None):
    # Calculate confusion matrix
    print('Confusion Matrix and Statistics\n\t   Prediction')
    if labels is None:
        labels = list(y_true.unique())
    cm = mtrs.confusion_matrix(y_true, y_pred, labels=labels, sample_weight=sample_weight, normalize=normalize)
    cm_df = pd.DataFrame(cm, columns=labels)
    cm_df = pd.DataFrame(labels, columns=['Reference']).join(cm_df)
    print(cm_df.to_string(index=False))
    if len(labels) == 2:
        # Calculate accuracy
        acc = mtrs.accuracy_score(y_true, y_pred, normalize=None, sample_weight=sample_weight)/len(y_true)
        # Calculate No Information Rate
        combos = np.array(np.meshgrid(y_pred, y_true)).reshape(2, -1)
        noi = mtrs.accuracy_score(combos[0], combos[1], normalize=None, sample_weight=sample_weight)/len(combos[0])
        # Calculate p-value Acc > NIR
        res = binom_test(cm.diagonal().sum(), cm.sum(), max(pd.DataFrame(cm).apply(sum,axis=1)/cm.sum()),'greater')
        # Calculate P-value mcnemar test
        MCN_pvalue = cont_tab.mcnemar(cm).pvalue
        # Calculate Kappa
        Kappa = mtrs.cohen_kappa_score(y_true, y_pred, labels=labels, sample_weight=sample_weight)
        # Calculate sensitivity, specificity et al
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)
        Pos_pred = TP / (TP + FP)
        Neg_pred = TN / (TN + FN)
        F_score = 2 * sens * Pos_pred / (sens + Pos_pred)
        Prevalence = (TP + FN) / (TP + TN + FP + FN)
        Detection_rate = TP / (TP + TN + FP + FN)
        Detection_prevalence = (TP + FP) /  (TP + TN + FP + FN)
        Balanced_acc = (sens + spec) / 2 
        Positive_class = labels[1]
        
        # print all the measures
        out_str = '\nAccuracy: ' + str(round(acc,2)) + '\n' + \
        'No Information Rate: ' + str(round(noi,2)) + '\n' + \
        'P-Value [Acc > NIR]: ' + str(round(res,2)) + '\n' + \
        'Kappa: ' + str(round(Kappa,2)) + '\n' + \
        'Mcnemar\'s Test P-Value: ' + str(round(MCN_pvalue,2)) + '\n' + \
        'Sensitivity: ' + str(round(sens,2)) + '\n' + \
        'Specificity: ' + str(round(spec,2)) + '\n' + \
        'Pos pred value: ' + str(round(Pos_pred,2)) + '\n' + \
        'Neg pred value: ' + str(round(Neg_pred,2)) + '\n' + \
        'Prevalence: ' + str(round(Prevalence,2)) + '\n' + \
        'Detection Rate: ' + str(round(Detection_rate,2)) + '\n' + \
        'Detection prevalence: ' + str(round(Detection_prevalence,2)) + '\n' + \
        'Balanced accuracy: ' + str(round(Balanced_acc,2)) + '\n' + \
        'F Score: ' + str(round(F_score,2)) + '\n' + \
        'Positive class: ' + Positive_class
        print(out_str)
    else:
        # Calculate accuracy
        acc = mtrs.accuracy_score(y_true, y_pred, normalize=None, sample_weight=sample_weight)/len(y_true)
        # Calculate No Information Rate
        combos = np.array(np.meshgrid(y_pred, y_true)).reshape(2, -1)
        noi = mtrs.accuracy_score(combos[0], combos[1], normalize=None, sample_weight=sample_weight)/len(combos[0])
        # Calculate p-value Acc > NIR
        res = binom_test(cm.diagonal().sum(), cm.sum(), max(pd.DataFrame(cm).apply(sum,axis=1)/cm.sum()),'greater')
        # Calculate P-value mcnemar test
        MCN_pvalue = cont_tab.mcnemar(cm).pvalue
        # Calculate Kappa
        Kappa = mtrs.cohen_kappa_score(y_true, y_pred, labels=labels, sample_weight=sample_weight)
        # Calculate sensitivity, specificity et al
        TP = cm[1,1]
        TN = cm[0,0]
        FP = cm[0,1]
        FN = cm[1,0]
        sens = TP / (TP + FN)
        spec = TN / (TN + FP)
        Pos_pred = TP / (TP + FP)
        Neg_pred = TN / (TN + FN)
        F_score = 2 * sens * Pos_pred / (sens + Pos_pred)
        Prevalence = (TP + FN) / (TP + TN + FP + FN)
        Detection_rate = TP / (TP + TN + FP + FN)
        Detection_prevalence = (TP + FP) /  (TP + TN + FP + FN)
        Balanced_acc = (sens + spec) / 2 
        Positive_class = labels[1]
        
        # print all the measures
        out_str = '\nAccuracy: ' + str(round(acc,2)) + '\n' + \
        'No Information Rate: ' + str(round(noi,2)) + '\n' + \
        'P-Value [Acc > NIR]: ' + str(round(res,2)) + '\n' + \
        'Kappa: ' + str(round(Kappa,2)) + '\n' + \
        'Mcnemar\'s Test P-Value: ' + str(round(MCN_pvalue,2)) + '\n' 
        print(out_str)

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates,
    in the correct format for LineCollection:
    an array of the form
    numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    return segments


def colorline(x, y, z=None, axes=None,
              cmap=plt.get_cmap('coolwarm'),
              norm=plt.Normalize(0.0, 1.0), linewidth=3, alpha=1.0,
              **kwargs):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    if isinstance(z, numbers.Real):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = matplotlib.collections.LineCollection(
        segments, array=z, cmap=cmap, norm=norm,
        linewidth=linewidth, alpha=alpha, **kwargs
    )

    if axes is None:
        axes = plt.gca()

    axes.add_collection(lc)
    axes.autoscale()

    return lc


def plot_roc(tpr, fpr, thresholds, subplots_kwargs=None,
             label_every=None, label_kwargs=None,
             fpr_label='False Positive Rate',
             tpr_label='True Positive Rate',
             luck_label='Luck',
             title='Receiver operating characteristic',
             **kwargs):

    if subplots_kwargs is None:
        subplots_kwargs = {}

    figure, axes = plt.subplots(1, 1, **subplots_kwargs)

    if 'lw' not in kwargs:
        kwargs['lw'] = 1

    axes.plot(fpr, tpr, **kwargs)

    if label_every is not None:
        if label_kwargs is None:
            label_kwargs = {}

        if 'bbox' not in label_kwargs:
            label_kwargs['bbox'] = dict(
                boxstyle='round,pad=0.5', fc='yellow', alpha=0.5,
            )

        for k in six.moves.range(len(tpr)):
            if k % label_every != 0:
                continue

            threshold = str(np.round(thresholds[k], 2))
            x = fpr[k]
            y = tpr[k]
            axes.annotate(threshold, (x, y), **label_kwargs)

    if luck_label is not None:
        axes.plot((0, 1), (0, 1), '--', color='Gray')

    lc = colorline(fpr, tpr, thresholds, axes=axes)
    figure.colorbar(lc)

    axes.set_xlim([-0.05, 1.05])
    axes.set_ylim([-0.05, 1.05])

    axes.set_xlabel(fpr_label)
    axes.set_ylabel(tpr_label)

    axes.set_title(title)

    axes.legend(loc="lower right")

    return figure, axes

def plotClassPerformance(y, prob_est, selClass):
    if len(y.unique()) == 2:
        if  len(prob_est.shape) > 1:
            if not prob_est.shape[1] == 1:
                prob_est = prob_est[:,y.unique() == selClass]
        # Calibration plot
        # Use cuts for setting the number of probability splits
        points_y, points_x = calibration_curve(y, prob_est, n_bins=10)
        fig, ax = plt.subplots()
        # reference line, legends, and axis labels
        line = mlines.Line2D([0, 1], [0, 1], color='black')
        transform = ax.transAxes
        line.set_transform(transform)
        ax.add_line(line)
        plt.plot(points_x, points_y, marker='o', linewidth=1, label='model')
        fig.suptitle('Plot 1/4: Calibration plot')
        ax.set_xlabel('Predicted probability')
        ax.set_ylabel('True probability in each bin')
        plt.legend()
        plt.show()
        
        # Probability histograms
        df = pd.DataFrame(y)
        df['prob_est'] = prob_est 
        sns.set(style ="ticks")
        d = {'color': ['r', 'b']}
        g = sns.FacetGrid(df, col='Y', hue='Y', hue_kws=d, margin_titles=True)
        bins = np.linspace(0, 1, 10)
        g.map(plt.hist, "prob_est", bins=bins)
        plt.subplots_adjust(top=0.8)
        g.fig.suptitle('Plot 2/4: Probability of Class ' + selClass) # can also get the figure from plt.gcf() 

        # calculate roc curve
        fpr, tpr, thresholds = mtrs.roc_curve(y, prob_est, pos_label=selClass)
        if tpr[1] - tpr[0] < fpr[1] - fpr[0]:
            fpr, tpr = tpr, fpr
        roc_auc = mtrs.auc(fpr, tpr)

        plot_roc(tpr, fpr, thresholds, title = 'Plot 3/4: ROC, Area under the ROC curve - ' + str(round(roc_auc,3)))

        y_true = y == selClass
        accuracy_scores = []
        for thresh in thresholds:
            accuracy_scores.append(mtrs.accuracy_score(y_true, [1 if m > thresh else 0 for m in prob_est]))
        
        accuracy_scores = np.array(accuracy_scores)

        fig, ax = plt.subplots()
        ax.plot(thresholds, accuracy_scores, color='navy')
        ax.set_xlim([0.0, 1.0])
        ax.set_title('Plot 4/4: Accuracy across possible cutoffs')

def dotplot(scores, metric:str):
    plt.xlabel(metric)
    plt.ylabel('')
    plt.title("Scores")
    scores_list = [score for key, score in scores.items()]
    for i in range(len(scores_list)):
        plt.boxplot(scores_list[i], positions=[i], vert=False)
    plt.yticks(list(range(len(scores_list))), list(scores.keys()))
    plt.show()

def calibration_plot(real, estimations):
    fig, ax = plt.subplots()
    for col in estimations.columns:
        y, x = calibration_curve(real, estimations.loc[:,col], n_bins=10)
        # only these two lines are calibration curves
        plt.plot(x, y, marker='o', linewidth=1, label=col)

    # reference line, legends, and axis labels
    line = mlines.Line2D([0, 1], [0, 1], color='black')
    transform = ax.transAxes
    line.set_transform(transform)
    ax.add_line(line)
    fig.suptitle('Calibration plot')
    ax.set_xlabel('Predicted probability')
    ax.set_ylabel('True probability in each bin')
    plt.legend()
    plt.show()

def roc_curve(real, estimations, selClass, fpr_label='False Positive Rate', tpr_label='True Positive Rate', title='Receiver operating characteristic (ROC)'):
    fig, ax = plt.subplots()
    for col in estimations.columns:
        fpr, tpr, thresholds = mtrs.roc_curve(real, estimations.loc[:,col], pos_label=selClass)
        # only these two lines are calibration curves
        ax.plot(fpr, tpr, linewidth=1, label=col)
        print('Area under the ROC curve of', col,':', round(mtrs.auc(fpr, tpr), 3))
    ax.plot((0, 1), (0, 1), '--', color='Gray')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    ax.set_xlabel(fpr_label)
    ax.set_ylabel(tpr_label)

    ax.set_title(title)

    ax.legend(loc="lower right")
    
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)