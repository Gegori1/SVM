class SVM_general():
    """ Classify binomial separable and non-separable data through linear and non_linear models.
    
        -- Parameter --
            C: determines the number of points that contribute to creating the boundary. 
               (Default = 0.1)
               The bigger the value of C, the lesser the points that the model will consider.
        
            kernel: name of the kernel that the model will use. Written in a format string 
                    (Default = "linear"). 
        
                    acceptable parameters: 
                        "linear", "poly", "polynomial", "rbf", 
                        "laplacian", "cosine".
        
                    for more information about individual kernels, visit the 
                    sklearn pairwise metrics affinities and kernels user guide.

        --Methods--
            fit(X, y): Learn from the data. Returns self.

            predict(X_test): Predicts new points. Returns X_test labels.

            coef_(): Returns linear model w and b coefficients, or w, x, and b
                     for non_linear models

            For more information about each method, visit specific documentations.
            
        --Example-- 
            ## Calls the classes in SVMLibrary_generalkernel notebook
            >>> from SVM import SVM_general
            ...
            ## Initializes the object with custom parameters
            >>> model = SVM_general(C = 0.01, kernel = "linear")
            ...
            ## Uses the model to fit the data
            >>> fitted_model = model.fit(X, y)
            ...
            ## Predicts with the given model
            >>> y_prediction = fitted_model.predict(X_test)
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
        """ Computes coefficients for the new data prediction.
        
            --Parameters--
                X: nxm matrix that contains all data points
                   components. n is the number of points and
                   m is the number of features of each point.
                   
                y: nx1 matrix that contain the labels for all
                   the points.
            
            --Returns--
                self, containing all the parameters needed to 
                compute new data points: x_k, w, b, and c and a
                which indicate the original label and the label
                used inside the model (e.g [0, 1]->[1, -1])
        """
        def preprocess(X, y):
            # label preprocessing
            a = np.unique(y); c = np.array([1, -1])
            y = np.where(y == a[0], c[0], c[1])

            # pre_matrices
            H = self.pairwise_kernels(X, X, metric = self.kernel); Y = np.outer(y, y)
            Q = np.multiply(Y, H) 
            q = -np.ones(y.shape)
            A = np.array(y.reshape(1, -1), dtype = "float64"); b = 0.0
            ydim = y.shape[0]
            G = np.concatenate((np.identity(ydim), -np.identity(ydim)))
            h_ = np.concatenate((self.C*np.ones(ydim), np.zeros(ydim))) 
            h = h_.reshape(-1, 1)
            # return self for later use on predict method
            self.c = c; self.a = a
            return {"H":H, "Q":Q, "q":q, "A":A, "b":b, "G":G, "h":h}
        
        def solver(pre):
            # matrices for the solver
            matrix = self.matrix
            Q = matrix(pre["Q"]); q = matrix(pre["q"])
            A = matrix(pre["A"]); b = matrix(pre["b"])
            G = matrix(pre["G"]); h = matrix(pre["h"])
            # solver
            solvers = self.solvers
            solvers.options['show_progress']=False
            sol=solvers.qp(P=Q, q=q,G=G,h=h, A=A, b=b)
            return sol
        
        def get_coefficients(sol, H):
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
            
            # keep on memory for leter use
            self.x_sv = x_sv; self.w = w; self.b = b
            
        # run all previous functions inside fit method
        def run():
            # preprocess the labels and 
            # get the prematrice
            param = preprocess(X, y)
            
            # solve for the alphas
            sol = solver(param)
            
            #compute the coefficients
            get_coefficients(sol, param["H"])
            
            return self
        
        
        return run()
    
    # predict
    def predict(self, X_test):
        """Predicts new labels for a given set of new 
           independent variables (X_test).
           
           --Parameters--
               X_test: nxm matrix containing all the points that 
                       will be predicted by the model.
                       n is the number of points. m represents the
                       number of features/dimensions of each point.
            
           --Returns--
               a nx1 vector containing the predicted labels for the 
               input variables.
               
               
           --Example-- 
                ## Calls the classes in SVMLibrary_generalkernel notebook
                >>> from SVM_ import SVM_general
                ...
                ## Initializes the object with custom parameters
                >>> model = SVM_general(C = 0.01, kernel = "linear")
                ...
                ## Uses the model to fit the data
                >>> fitted_model = model.fit(X, y)
                ...
                ## Predicts with the given model
                >>> y_prediction = fitted_model.predict(X_test)
                ...
                ## e.g
                >>> print(y_prediction)
                np.array([1, 1, 1, 0, 0, 1, 0])
                
        """
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