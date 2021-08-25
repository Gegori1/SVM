# -*- coding: utf-8 -*-
"""
@author: rodrigsa
"""


import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MaxAbsScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import check_random_state
from sklearn.datasets import load_boston
import pandas as pd
from pathlib import Path
import cvxpy as cp

from bayes_opt import BayesianOptimization
from bayes_opt import SequentialDomainReductionTransformer
from bayes_opt import UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from gplearn.genetic import SymbolicTransformer

#%%MAPE Extendido

class SVR_cvxopt_mapext:
    
    def __init__(self, C = 0.1, epsilon = 0.01, lamda = 0.2, kernel = "linear", **kernel_param):
        import numpy as np
        from cvxopt import matrix, solvers, sparse
        from sklearn.metrics.pairwise import pairwise_kernels
        from sklearn.utils import check_X_y, check_array 
        self.sparse = sparse
        self.matrix = matrix
        self.solvers = solvers
        self.C = C
        self.epsilon = epsilon
        self.kernel = kernel
        self.pairwise_kernels = pairwise_kernels
        self.kernel_param = kernel_param
        self.check_X_y = check_X_y
        self.check_array = check_array
        self.lamda = lamda
        
    def fit(self, X, y):
        X, y = self.check_X_y(X, y)
        # hyperparameters
        C = self.C 
        epsilon =  self.epsilon
        lamda = self.lamda
        
        kernel = self.kernel
        pairwise_kernels = self.pairwise_kernels
        
        sparse = self.sparse 
        matrix = self.matrix 
        solvers = self.solvers 
        
        # Useful parameters
        ydim = y.shape[0]
        onev = np.ones((ydim,1))
        x0 = np.random.rand(ydim)
        
        # Prematrices for the optimizer
        K = pairwise_kernels(X, X, metric = kernel, **self.kernel_param)
        A = onev.T
        b = 0.0
        G = np.concatenate((np.identity(ydim), -np.identity(ydim)))
        h_ = np.concatenate((100*C*np.ones(ydim)/y, 100*C*np.ones(ydim)/y)); 
        h = h_.reshape(-1, 1)

        # Matrices for the optimizer
        A = matrix(A)
        b = matrix(b)
        G = sparse(matrix(G))
        h = matrix(h)
        Ev = (epsilon*y.T)/100
        
        # functions for the optimizer
        def obj_func(x):
            return 0.5* x.T @ K @ x - y.T @ x + lamda*((1-Ev)@np.abs(x) + Ev/2 @ x)

        def obj_grad(x):
            return x.T @ K + lamda*((1 - Ev) @ (x/np.abs(x)) + Ev/2) - y
        
        def F(x = None, z = None):
            if x is None: return 0, matrix(x0)
            # objective dunction
            val = matrix(obj_func(x))
            # obj. func. gradient
            Df = matrix(obj_grad(x))
            if z is None: return val, Df
            # hessian
            H = matrix(z[0] * K)
            return val, Df, H
        
        # Solver
        solvers.options['show_progress'] = False
        sol = solvers.cp(F=F, G=G, h=h, A=A, b=b)
        
        # Support vectors
        beta_1 = np.array(sol['x']).reshape(-1)
        beta_n = np.abs(beta_1)/beta_1.max()
        indx = beta_n > 1e-4
        beta_sv = beta_1[indx]
        x_sv = X[indx,:]
        y_sv = y[indx]
        
        # get w_phi and b
        k_sv = pairwise_kernels(x_sv, x_sv, metric = kernel, **self.kernel_param)
        cons = np.where(beta_sv >= 0, 1 - epsilon/100, 1 + epsilon/100)
        
        w_phi = beta_sv @ k_sv
        b = np.mean((y_sv*cons - w_phi)); self.b = b
        self.beta_sv = beta_sv; self.x_sv = x_sv
        return self
        
    def predict(self, X_):
        X_ = self.check_array(X_)
        k_test = self.pairwise_kernels(self.x_sv, X_, metric = self.kernel, **self.kernel_param)
        w_phi_test = self.beta_sv @ k_test
        predict = w_phi_test + self.b
        return predict
    
    def coef_(self):
        return self.beta_sv, self.x_sv, self.b
    
    
#%%Load Data
rng = check_random_state(0)
boston = load_boston()
perm = rng.permutation(boston.target.size)
boston.data = boston.data[perm]
boston.target = boston.target[perm]

Y = boston.target
X = boston.data    

#Symbolic Transformer
function_set = ['add', 'sub', 'mul', 'div', 'sqrt', 'log',
                'abs', 'neg', 'inv', 'max', 'min']
model = SymbolicTransformer(generations=20, population_size=2000,
                         hall_of_fame=100, n_components=10,
                         function_set=function_set,
                         parsimony_coefficient=0.0005,
                         max_samples=0.9, verbose=1,
                         random_state=0)

model.fit(X, Y)
model_params = model.get_params()
gp_features = model.transform(X)
new_data = np.hstack((X, gp_features))

X = new_data
X = np.array(X)
X = X.astype(float)
#split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state = 5)

#normalize
scaler = MaxAbsScaler().fit(x_train)
X_train = scaler.transform(x_train); X_test = scaler.transform(x_test)

scaler1 = MaxAbsScaler().fit(y_train.reshape(-1, 1))
Y_train = scaler1.transform(y_train.reshape(-1, 1)).reshape(-1)
Y_test = scaler1.transform(y_test.reshape(-1, 1)).reshape(-1)   


#%%
def opt_bas(C, epsilon, lamda, gamma):
    
    # parameters
    hyperparameters = {
        'kernel' : "rbf",
        'C' : C, 
        'epsilon' : epsilon, 
        'lamda' : lamda,
        'gamma' : gamma,
    }
    
    # fit and predict
    model = SVR_cvxopt_mapext(**hyperparameters).fit(X_train, Y_train)

    predict = model.predict(X_test)
    
    # rescale
    y_pred = scaler1.inverse_transform(predict.reshape(-1, 1)).reshape(-1)
    
    # get score
    mape = np.mean(np.abs((y_test - y_pred)/y_test))*100
    
    return -mape 


#%%
class newJSONLogger(JSONLogger):

      def __init__(self, path):
            self._path=None
            super(JSONLogger, self).__init__()
            self._path = path if path[-5:] == ".json" else path + ".json"    
            
            
#%%
# Bounded region of parameter space
pbounds = {'C': (0.1, 10), 'epsilon': (0.001, 10), 'lamda': (0.1, 1), 'gamma': (0.01, 2)}

# Domain reduction function
# bounds_transformer = SequentialDomainReductionTransformer()

# Bayes optimizer instantiation
optimizer = BayesianOptimization(f=opt_bas, 
                                 pbounds=pbounds, 
                                 random_state=1, verbose=2, 
#                                  bounds_transformer=bounds_transformer
                                )

# keep data
log_path = Path().resolve() / "Logs" / "onepoint.json"
logger = newJSONLogger(path = str(log_path))
optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)

#%%

#optimizer.probe(
#    params={"C": 0.1, "epsilon": 0.01, "lamda": 0.2, "gamma": 0.3},
#    lazy=True,
#)

#%%

optimizer.maximize(init_points=1, n_iter=200)

#%%
c = optimizer.max['params']['C']
gamma = optimizer.max['params']['gamma']
e = optimizer.max['params']['epsilon']
lamda = optimizer.max['params']['lamda']

#%%
print(optimizer.max)