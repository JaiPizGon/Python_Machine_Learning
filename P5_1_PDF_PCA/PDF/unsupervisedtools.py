import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
def biplot(score,coeff,y,labels=None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    scale_xs = xs * scalex
    scale_ys = ys * scaley
    # df = pd.DataFrame({'x':xs * scalex, 'y':ys * scaley})
    # df['c'] = y.values
    fig, ax = plt.subplots()
    # if labels is None:
    #     sns.scatterplot(x='x', y='y', data=df)
    # else:
    #     sns.scatterplot(x='x', y='y', hue='c', data=df)
    for i, txt in enumerate(y.values):
        ax.annotate(txt[0], (scale_xs[i], scale_ys[i]))
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid()