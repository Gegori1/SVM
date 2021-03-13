class SVM():
    """ Classify binomial separable and non-separable data through linear and non_linear models.
    
        -- Parameter --
            C: determines the number of points that contribute to creating the boundary. 
                (Default = 0.1)
                The bigger the value of C, the lesser the points that the model will consider.
        
            kernel: name of the kernel that the model will use. Written in a format string 
                (Default = "linear"). 
        
                acceptable parameters: 
                    "additive_chi2", "chi2", "linear", "poly", 
                    "polynomial", "rbf", "laplacian", "sigmoid", "cosine".
        
                for more information about individual kernels, visit the 
                sklearn pairwise metrics affinities and kernels user guide.

        --Methods--
            fit(X, y): Learn from the data. Returns self.

            predict(X_test): Predicts new points. Returns X_test labels.

            coef_(): Returns linear model w and b coefficients, or w, x, and b
                for non_linear models

            For more information about each method, visit specific documentations.
            
        --Example-- 
            ## Calls the classes in SVC_library_non_linear_kernel notebook
            >>> from SVM_ import SVM
            ...
            ## Initializes the object with custom parameters
            >>> model = SVM(C = 0.01, kernel = "linear")
            ...
            ## Uses the model to fit the data
            >>> fitted_model = model.fit(X, y)
            ...
            ## Predicts with the given model
            >>> y_prediction = fitted_model(X_test)
            ...
            ## e.g
            >>> print(y_prediction)
            np.array([1, 1, 1, 0, 0, 1, 0])
    """
    
    def __init__(self, C=0.1, kernel = "linear"):
        from sklearn.metrics.pairwise import pairwise_kernels
        from cvxopt import solvers, matrix
        self.C = C
        self.pairwise_kernels = pairwise_kernels
        self.kernel = kernel
        self.matrix = matrix
        self.solvers = solvers
        
    # learn   
    def fit(self, X, y):
        
        # label preprocessing
        a = np.unique(y); c = np.array([1, -1])
        y = np.where(y == a[0], c[0], c[1])
        
        # pre_matrices
        H = self.pairwise_kernels(X, X, metric = self.kernel); Y = np.outer(y, y)
        Q = np.multiply(Y, H); q = -np.ones(y.shape)
        A = np.array(y.reshape(1, -1), dtype = "float64"); b = 0.0
        ydim = y.shape[0]
        G = np.concatenate((np.identity(ydim), -np.identity(ydim)))
        h_ = np.concatenate((self.C*np.ones(ydim), np.zeros(ydim))); h = h_.reshape(-1, 1)
        
        # matrices for the solver
        matrix = self.matrix
        Q = matrix(Q); q = matrix(q)
        A = matrix(A); b = matrix(b)
        G = matrix(G); h = matrix(h)
        # solver
        solvers = self.solvers
        solvers.options['show_progress']=False
        sol=solvers.qp(P=Q, q=q,G=G,h=h, A=A, b=b)
        
        # alphas threshhold and svs
        alphas = np.array(sol['x']); indx = alphas > 1e-10 
        alpha_sv = alphas[indx]
        h_sv = H[indx[:,0],:]; y_sv = y[indx[:,0]]; x_sv = X[indx[:,0],:]

        # a_k * y_k * K_k
        w = (alpha_sv*y_sv).reshape(-1, 1)
        ayk = np.multiply(w, h_sv)
        # w and b
        w_phi = np.sum(ayk, axis=0)
        b = np.mean(y-w_phi)
        
        self.x_sv = x_sv
        self.w = w; self.b = b; self.c = c; self.a = a
        return self
    
    # predict
    def predict(self, X_test):
        # rename label variables
        c = self.c; a = self.a
        # rename coefficients
        b = self.b; x_sv = self.x_sv
        
        # create new kernel
        H = self.pairwise_kernels(x_sv, X_test, metric = self.kernel)
        # multiply w and kernel
        w_phi = np.sum(np.multiply(self.w, H), axis = 0)
        
        # predict new data
        predict1 = np.sign(w_phi + b)
        # rename to original labels
        predict2 = np.where(predict1 == c[0], a[0], a[1])
        return predict2
    
    # coefficient
    def coef_(self):
        if self.kernel == "linear":
            w = self.w; x_sv = self.x_sv
            w = np.sum(np.multiply(w, x_sv), axis = 0)
            return w, self.b
        else: 
            return self.w, self.x_sv,  self.b