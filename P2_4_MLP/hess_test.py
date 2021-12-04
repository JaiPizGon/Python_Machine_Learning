import dill
import neuralsens as ns 
import numpy as np
from sklearn import set_config
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
set_config(display='diagram')

filepath = 'session.pkl'
dill.load_session(filepath) # Load the session


X = X_df.loc[:, ["X1","X2","X3","X4","X5"]]
y = 1 * X_df["X1"] + 1.5 * X_df["X2"] ** 2 + 0.5 * X_df["X3"] * X_df["X4"] + X_df["X4"] ** 3 
## Divide the data into training and test sets ---------------------------------------------------
## Create random 80/20 % split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

param = {'MLP__learning_rate_init': [0.01,0.1], # Initial value of learning rate 
         'MLP__hidden_layer_sizes':[(20,),(25,)]} # Number of neurons in each hidden layer, enters as tuples
pipe = Pipeline([
            ('MLP', MLPRegressor(solver='adam', # Update function
                                activation='logistic', # Logistic sigmoid activation function
                                alpha=0.01, # L2 regularization term
                                learning_rate='adaptive', # Type of learning rate used in training
                                max_iter=250, # Maximum number of iterations
                                batch_size=10, # Size of batch when training
                                tol=1e-4, # Tolerance for the optimization
                                validation_fraction=0.0, # Percentage of samples used for validation
                                n_iter_no_change=10, # Maximum number of epochs to not meet tol improvement
                                random_state=150))
        ])
# We use Grid Search Cross Validation to find the best parameter for the model in the grid defined 
nFolds = 10
MLP_fit = GridSearchCV(estimator=pipe, # Structure of the model to use
                       param_grid=param, # Defined grid to search in
                       n_jobs=-1, # Number of cores to use (parallelize)
                       scoring='neg_mean_squared_error', # RMSE https://scikit-learn.org/stable/modules/model_evaluation.html
                       cv=nFolds) # Number of Folds 
MLP_fit.fit(X_train, y_train) # Search in grid

mlp_predict = MLP_fit.predict(X_train)

correlation_matrix = np.corrcoef(mlp_predict, y_train)
correlation_xy = correlation_matrix[0,1]
r_squared = correlation_xy**2

print("Accuracy of MLP:", r_squared)

y = y_train
y = pd.DataFrame(y, columns={"Y"})
y.name = 'Y'
mlp = MLP_fit.best_estimator_['MLP']
wts = mlp.coefs_
bias = mlp.intercepts_
actfunc = ['identity',MLP_fit.best_estimator_['MLP'].get_params()['activation'],mlp.out_activation_]
#X = MLP_fit.best_estimator_['preprocessor'].transform(X_train) # Preprocess the variables
if X_train.select_dtypes('category').shape[1] > 0:
    column_init = X_train.select_dtypes(exclude='category').shape[1]
    input_names = []
    for cat_input in X_train.select_dtypes('category').columns:
        input_names += [cat_input + str(cat) for cat in X_train[cat_input].unique()]
    coefnames = X_train.select_dtypes(exclude='category').columns.values.tolist() + input_names
else:
    coefnames = X_train.columns.values.tolist()

X = X_df
sens_end_layer = 'last'
sens_end_input = False
sens_origin_layer = 0
sens_origin_input = True
dill.dump_session(filepath) # Load the session
sensmlp = ns.SensAnalysisMLP(wts, bias, actfunc, X_df, y)

sensmlp.summary()

sensmlp.info()

sensmlp.plot()

hessmlp = ns.HessianMLP(wts, bias, actfunc, X_df, y)

hessmlp.summary()

hessmlp.info()

hessmlp.plot()