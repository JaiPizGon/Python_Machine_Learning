from scipy.special import expit
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
import pandas as pd
import numbers

def ActivationFunction(func):
    def identity(x):
        return x
    def relu(x):
        return x * (x > 0)
    actfunc = {
        'logistic': expit,
        'identity': identity,
        'tanh': np.tanh,
        'relu': relu
        }
    return actfunc[func]

def DerActivationFunction(func:str):
    def sigmoid(x):
        fx = expit(x)
        return np.diag(fx * (1 - fx))
    def identity(x):
        return np.identity(x.shape[0], dtype=int)
    def tanh(x):
        return np.diag(1 - np.tanh(x) ** 2)
    def relu(x):
        return np.diag(x > 0) 
    deractfunc = {
        'logistic': sigmoid,
        'identity': identity,
        'tanh': tanh,
        'relu': relu
        }
    return deractfunc[func]

def SensAnalysisMLP(wts, bias, actfunc, X, y, 
                    sens_origin_layer=0, sens_end_layer='last', sens_origin_input=True, sens_end_input=False):
    ### Initialize all the necessary variables
    # Structure of the mlp model
    mlpstr = [wts[0].shape[0]] + [lyr.shape[1] for lyr in wts] 

    # Derivative and activation functions for each neuron layer
    deractfunc = [DerActivationFunction(af) for af in actfunc]
    actfunc = [ActivationFunction(af) for af in actfunc]

    # Weights of input layer
    W = [np.identity(X.shape[1])]

    # Input of input layer 
    # inputs = [np.hstack((np.ones((len(X_train),1), dtype=int), X_train))]
    Z = [np.dot(X, W[0])]

    # Output of input layer
    O = [actfunc[0](Z[0])]

    # Derivative of input layer
    D = [np.array([deractfunc[0](Z[0][irow,]) for irow in range(Z[0].shape[0])])]

    # Let's go over all the layers calculating each variable
    for lyr in range(1,len(mlpstr)):
        # Calculate weights of each layer
        W.append(np.vstack((bias[lyr-1], wts[lyr-1])))
        # Calculate input of each layer
        # Add columns of 1 for the bias
        aux = np.ones((O[lyr-1].shape[0],O[lyr-1].shape[1]+1))
        aux[:,1:] = O[lyr-1]
        Z.append(np.dot(aux,W[lyr]))
        # Calculate output of each layer
        O.append(actfunc[lyr](Z[lyr]))
        # Calculate derivative of each layer
        D.append(np.array([deractfunc[lyr](Z[lyr][irow,]) for irow in range(X.shape[0])]))

    # Now, let's calculate the derivatives of interest
    if sens_end_layer == 'last':
        sens_end_layer = len(actfunc)

    warn = ''' if not all(isinstance([sens_end_layer, sens_origin_layer], numbers.Number)):
        pass # Warning explaining that they should send a number in the layers

    if any([sens_end_layer, sens_origin_layer] <= 0):
        pass # Warning explaining that the number of layers should be positive

    if not((sens_end_layer > sens_origin_layer) or ((sens_end_layer == sens_origin_layer) and (sens_origin_input and not sens_end_input))):
        pass # Warning explaining that at least one layer of neurons must exist between end and origin

    if any([sens_end_layer, sens_origin_layer] > len(actfunc)):
        pass # Warning explaining that layers specified could not be found in the model'''

    D_accum = [np.identity(mlpstr[sens_origin_layer]) for irow in range(X.shape[0])]
    if sens_origin_input:
        D_accum = [D[sens_origin_layer]]

    counter = 0
    # Only perform further operations if origin is not equal to end layer  
    if not (sens_origin_layer == sens_end_layer):
        for layer in range(sens_origin_layer + 1, sens_end_layer):
            counter += 1
            # Calculate the derivatives of the layer based on the previous and the weights
            if (layer == sens_end_layer) and sens_end_input:
                D_accum.append(np.array([np.dot(D_accum[counter - 1][irow,], W[layer][1:,:]) for irow in range(X.shape[0])]))
            else:
                D_accum.append(np.array([np.dot(np.dot(D_accum[counter - 1][irow,], W[layer][1:,]), D[layer][irow,]) for irow in range(X.shape[0])]))
    
    # Calculate sensitivity measures for each input and output 
    meanSens = np.mean(D_accum[counter], axis=0)
    stdSens = np.std(D_accum[counter], axis=0)
    meansquareSens = np.mean(np.square(D_accum[counter]), axis=0)
    
    # Store the information extracted from sensitivity analysis
    input_name = X.columns.values.tolist()
    sens = [pd.DataFrame({'mean': meanSens[:,icol], 'std': stdSens[:,icol], 'mean_squared': meansquareSens[:,icol]}, index=input_name) for icol in range(meanSens.shape[1])]
    raw_sens = [pd.DataFrame(D_accum[counter][:,:,out], index=range(X.shape[0]), columns=input_name) for out in range(D_accum[counter].shape[2])]
    
    # Create output name for creating self
    output_name = y.name
    if D_accum[counter].shape[2] > 1:
        output_name =  ['_'.join([y.name, lev]) for lev in y.unique()]
    return self(sens, raw_sens, mlpstr, X, input_name, output_name)

# Define self class
class self:
    def __init__(self, sens, raw_sens, mlp_struct, X, input_name, output_name):
        self.__sens = sens
        self.__raw_sens = raw_sens
        self.__mlp_struct = mlp_struct
        self.__X = X
        self.__input_names = input_name
        self.__output_name = output_name
    @property
    def sens(self):
        return self.__sens
    @property
    def raw_sens(self):
        return self.__raw_sens
    @property
    def mlp_struct(self):
        return self.__mlp_struct
    @property
    def X(self):
        return self.__X
    @property
    def input_name(self):
        return self.__input_name
    @property
    def output_name(self):
        return self.__output_name

    def summary(self):
        print("Sensitivity analysis of", str(self.mlp_struct), "MLP network.\n")
        print("Sensitivity measures of each output:\n")
        for out in range(len(self.sens)):
            print("$" + self.output_name[out], "\n")
            print(self.sens[out])
    
    def info(self, n=5):
        print("Sensitivity analysis of", str(self.mlp_struct), "MLP network.\n")
        print(self.X.shape[0],'samples\n')
        print("Sensitivities of each output (only ",min([n,self.raw_sens[0].shape[0]])," first samples):\n", sep = "")
        for out in range(len(self.raw_sens)):
            print("$" + self.output_name[out], "\n")
            print(self.raw_sens[out][:min([n,self.raw_sens[out].shape[0]])])

    def plot(self, type='sens'):
        if type=='sens':
            self.sensitivityPlots()
        elif type=='features':
            self.featurePlots()
        elif type=='time':
            self.timePlots()
        else:
            print('The specifyed type', type, 'is not an accepted plot type.')
    def sensitivityPlots(self):
        pltlist = []
        for out in range(len(self.raw_sens)):
            sens = self.sens[out].sort_values(by='mean_squared')
            raw_sens = self.raw_sens[out]
            # Plot mean-std plot
            fig, ax = plt.subplots()
            ax.set_xlim([min(sens['mean']) - 0.2 * max(abs(sens['mean'])), max(sens['mean']) + 0.2 * max(abs(sens['mean']))])
            ax.hlines(y=0, xmin=min(sens['mean']) - 0.2 * max(abs(sens['mean'])), xmax=max(sens['mean']) + 0.2 * max(abs(sens['mean'])), color='blue')
            ax.vlines(x=0, ymin=0, ymax=max(abs(sens['std'])), color='blue')
            ax.scatter(x=0, y=0, s=150, c='blue')
            for i, txt in enumerate(sens.index.values.tolist()):
                ax.annotate(txt, xy=(sens['mean'][i], sens['std'][i]),  xycoords='data', va="center", ha="center", fontsize='large',
                        bbox=dict(boxstyle="round", fc="w", ec="gray"))
            ax.set_xlabel('mean(Sens)')
            ax.set_ylabel('std(Sens)')
            ax.plot()

            # Plot variable importance mean_sqaured
            colors = plt.cm.cmap_d['Blues_r'](sens['mean_squared']*0.5/max(sens['mean_squared']))
            ax2 = sens.plot.bar(y='mean_squared', color=colors, legend=False, rot=0)
            ax2.set_xlabel('Input variables')
            ax2.set_ylabel('mean(Sens^2)')
            ax2.plot()

            # Plot density of sensitivities
            ax3 = raw_sens.plot.kde()
            ax3.set_xlabel('Sens')
            ax3.set_ylabel('density(Sens)')
            ax3.plot()
    def featurePlots(self):
        pass
    def timePlots(self):
        pass