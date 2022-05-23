import numpy as np
from numpy import linalg as LA
from time import time
from abc import abstractmethod
from math import sqrt
from scipy.sparse import csr_matrix

class Solver:
    """ Abstract class representing a solver for the 
        for the linearly constrained convex program
        mininimize   f(x)
        subject to   Bx = b 
                    x >= 0,
        with variable x. 

       B is m x d.       
       The matrix A (which is n x d) is a data matrix.
    """

    def __init__(self, B, b, alpha_backtrack, beta_backtrack):

        self.B, self.b = B, b  
        self.hess = []
        self.alpha_backtrack = alpha_backtrack
        self.beta_backtrack = beta_backtrack

    
    # Child classes must implement evaluate_hessian. The function should modify
    # the variable self.hess.
    @abstractmethod
    def evaluate_hessian(self, x):
        pass

     # Child classes must implement evaluate_grad. The function should
     # return the gradient as a numpy array.
    @abstractmethod
    def evaluate_grad(self, x):
        pass

    @abstractmethod
    def evaluate_grad_precomputed(self, Ax, Adx, alpha):
        pass

    # nu is the multiplier corresponding to the equality constraint Bx = b.
    # _lambda is the mulitplier corresponding to the inequality constraint x>=0.
    # OBS: This line search is not for a general problem. It has been modified to the special case of logistic regression with
    # ell-1 regularization.
    def backtracking(self, x, _lambda, nu, dx, dlambda, dnu, t, original_residual_norm):
        indices_1 = np.where(dlambda < 0)[0]
        if indices_1.size == 0:
            alpha_max_1 = 1 
        else:
            alpha_max_1 = np.min((1, np.min(-_lambda[indices_1]/dlambda[indices_1])))
        
        indices_2 = np.where(dx < 0)[0]
        if indices_2.size == 0:
            alpha_max_2 = 1 
        else: 
            alpha_max_2 =np.min((1, np.min(-x[indices_2]/dx[indices_2])))

        alpha =  0.99 * np.min((alpha_max_1, alpha_max_2))
        
        max_backtracking_steps = 10 
        iter = 0

        # Some precomputations for efficient line search, see convex optimization book.
        temp1 = np.hstack((x[0:self.original_d] - x[self.original_d:2*self.original_d],
                           dx[0:self.original_d] - dx[self.original_d:2*self.original_d]))
        cache1 = self.yA @ temp1 

    
        while iter < max_backtracking_steps:
            iter = iter + 1
            x_plus = x + alpha*dx 
            lambda_plus = _lambda + alpha*dlambda 
            nu_plus = nu + alpha*dnu 

            grad = self.evaluate_grad_precomputed(cache1[:, 0], cache1[:, 1], alpha)
            norm_res = sqrt(LA.norm(grad - lambda_plus + self.B.T @ nu_plus)**2 + 
                            LA.norm(np.multiply(lambda_plus, x_plus) - 1/t)**2 + 
                            LA.norm(self.B @ x_plus - self.b)**2)
            if norm_res <= (1-self.alpha_backtrack*alpha)*original_residual_norm:
                break
            alpha = self.beta_backtrack*alpha

        return alpha


    """
    x0 = intial primal iterate, _lambda0 = multiplier for inequality constraints x>=0,
    nu = multiplier for equality constraints Bx = b.

    Note that x0 and _lambda0 must be strictly positive. 
    """
    def solve(self, max_iter, x0, _lambda0, nu0, mu):
        m, d = self.B.shape                              # Local d.
        x, _lambda, nu = x0, _lambda0, nu0 

        # Define parameters for tracking the progress.
        times, residual_norms, residual_opt_conds = [], [], []
        alphas, rd = np.zeros((max_iter, )), np.zeros((max_iter, ))
        rcent, rfeas = np.zeros((max_iter, )), np.zeros((max_iter, ))
        tic = time()

        # Iterations.
        for iter in range(0, max_iter):
            if(iter%10 == 0):
                print("Iteration: ", iter)
            times.append(time() - tic)
            nu_hat = (x.T @ _lambda)[0, 0]
            t = mu*d/nu_hat

            grad = self.evaluate_grad(x) 
            self.evaluate_hessian(x)

            # Should be possible to refactor this if it is the bottleneck.
            # Profiling the code suggests that the time for executing this part of the code is 
            # totally negligible, so no need for refactoring.
            row1 = np.concatenate((self.hess, -np.eye(d), self.B.T), axis = 1)
            row2 = np.concatenate((np.diag(_lambda.reshape(d,)), np.diag(x.reshape(d,)), np.zeros((d, m))), axis = 1)
            row3 = np.concatenate((self.B, np.zeros((m, d)), np.zeros((m, m))), axis = 1)
            coeff_matrix = np.concatenate((row1, row2, row3), axis = 0)

            _rp = self.B @ x - self.b
            _rd = grad - _lambda + self.B.T @ nu
            _rcent = np.multiply(_lambda, x) - 1/t
            residuals = np.concatenate((_rd, _rcent, _rp), axis = 0)
            
            rd[iter] = (LA.norm(residuals[0:d]))
            rcent[iter] = (LA.norm(residuals[d:2*d]))             
            rfeas[iter] = (LA.norm(residuals[2*d:]))
                                        
            residual_opt_cond = np.sqrt(LA.norm(_rp)**2 + LA.norm(_rd)**2 + nu_hat**2) 
            print("Iteration: ", iter, ". Norm residual: ", residual_opt_cond)
            residual_norms.append((LA.norm(residuals)))
            residual_opt_conds.append((residual_opt_cond))
            if(residual_opt_cond) < 10**(-7):
                break
           

            dir = -LA.solve(coeff_matrix, residuals)
            dx = dir[0:d]
            dlambda = dir[d:2*d]
            dnu = dir[2*d:]
        

            alpha = self.backtracking(x, _lambda, nu, dx, dlambda, dnu, t, LA.norm(residuals))      
            alphas[iter] = alpha

            x = x + alpha*dx
            _lambda = _lambda + alpha*dlambda 
            nu = nu + alpha*dnu 

        obj = self.compute_obj(x)
        return x, _lambda, nu, residual_norms, times, alphas, obj, rd, rcent, rfeas, residual_opt_conds




class Logistic_regression_L1_regularization(Solver):
    """ 

    """
    def __init__(self, B, b, gamma, A, y, alpha_backtrack, beta_backtrack,
                 sketching_size, sketching_strategy):
        super().__init__(B, b, alpha_backtrack, beta_backtrack)                                                       
        self.n, self.original_d = A.shape                                                # Number of data points.
        self.A = A 
        self.sketching_size = sketching_size 
        self.sketching_strategy = sketching_strategy
        self.y = y                                                         # Must be a column vector with one dimension (not zero).
        self.yA = self.y * self.A 
        self.gamma = gamma

    def evaluate_grad(self, w):
        x_plus = w[0:self.original_d]
        x_minus = w[self.original_d:2*self.original_d]
        #cached_quantity1_cmp = self.A.T * self.y.reshape(len(self.y), )
        cached_quantity1 = np.exp(self.yA @ (x_plus - x_minus))
        dh = np.divide(cached_quantity1, 1 + cached_quantity1)
        #grad_cmp = 1/self.n*np.vstack((cached_quantity1_cmp @ dh, -cached_quantity1_cmp @ dh,
        #                           np.zeros((self.r, 1)), np.zeros((self.r, 1))))
        #cached_quantity2 = self.A.T @ (self.y * dh)
        cached_quantity2 = self.yA.T @ dh
        term1 = 1/self.n*np.vstack((cached_quantity2, -cached_quantity2, np.zeros((3*self.original_d, 1))))
        term2 = np.zeros((5*self.original_d, 1))
        term2[2*self.original_d:3*self.original_d] = self.gamma*np.ones((self.original_d, 1))
        grad = term1 + term2
        return grad
    
    def evaluate_grad_precomputed(self, Ax, Adx, alpha):
        # First argument corresponds to y*A(x_plus - x_minus). Second argument corresponds to
        # y*A(dx_plus - dx_minus).

        cached_quantity1 = np.exp(Ax + alpha*Adx)
        dh = np.divide(cached_quantity1, 1 + cached_quantity1)
        #grad_cmp = 1/self.n*np.vstack((cached_quantity1_cmp @ dh, -cached_quantity1_cmp @ dh,
        #                           np.zeros((self.r, 1)), np.zeros((self.r, 1))))
        cached_quantity2 = self.A.T @ (self.y * dh.reshape((len(dh), 1)))
        term1 = 1/self.n*np.vstack((cached_quantity2, -cached_quantity2, np.zeros((3*self.original_d, 1))))
        term2 = np.zeros((5*self.original_d, 1))
        term2[2*self.original_d:3*self.original_d] = self.gamma*np.ones((self.original_d, 1))
        grad = term1 + term2

        return grad 

    def evaluate_hessian(self, w):

        cached_quantity1 = self.yA @ (w[0:self.original_d] - w[self.original_d:2*self.original_d])
        dsqrt = np.divide(np.exp(cached_quantity1/2), np.exp(cached_quantity1) + 1)

        # No sketching.
        if self.sketching_strategy == 1:
            stored_quantity = dsqrt * (self.yA)

        # COUNTSKETCH with numpy.
        elif self.sketching_strategy == 3: 
            indices = np.vstack([np.random.choice(self.sketching_size, self.n).reshape((1,-1)), np.arange(self.n)]) 
            values = np.random.choice(np.array([-1,1], dtype=np.float64), size=self.n)
            S = csr_matrix((values, indices), (self.sketching_size, self.n))

            # Sketch the square root of the Hessian.
            stored_quantity = S @ (dsqrt * (self.yA))


        cached_quantity2 = 1/self.n*stored_quantity.T @ stored_quantity
        row = np.hstack((cached_quantity2, -cached_quantity2, np.zeros((self.original_d, 3*self.original_d))))
        self.hess = np.vstack((row, -row, np.zeros((3*self.original_d, 5*self.original_d))))
        
    def compute_obj(self, w):
        x_original = w[0:self.original_d] - w[self.original_d:2*self.original_d]
        obj = np.sum(np.log((np.exp(self.A @ x_original) + 1)))
        return obj
 







def build_logistic_regression_ell1_regularization_problem_instance(n, d, rho):


    # Generate labels and data matrix.
    y = np.zeros((n, 1))
    for i in range(0, n):
        y[i] = 1 if np.random.random() < 0.5 else -1

    # Generate data matrix A.
    covariance_matrix = np.zeros((d, d))
    for i in range(0, d):
        for j in range(0, d):
            covariance_matrix[i, j] = 2*(rho**np.abs(i-j))

    A = np.random.multivariate_normal(np.zeros((d, )), covariance_matrix, size=n)


    # Build constraint matrix.
    B = np.zeros((2*d, 5*d))
    b = np.zeros((2*d, 1))
    
    B[0:d, 0:d]      = np.eye(d)
    B[0:d, d:2*d]    = -np.eye(d)
    B[0:d, 2*d:3*d]  = -np.eye(d)
    B[0:d, 3*d:4*d]  = np.eye(d) 

    B[d:, 0:d]      = -np.eye(d)
    B[d:, d:2*d]    = np.eye(d)
    B[d:, 2*d:3*d]  = -np.eye(d)
    B[d:, 4*d:]     = np.eye(d)
 
    return A, y, B, b 













































































class Logistic_regression_fairness_constraints(Solver):
    """ Solver for the problem
        minimize f(x) = 1/n \sum_{i=1}^n log(1+exp(y_i a_i^T x))
        subject to
            (1/n \sum_{i=1}^n (z_i - \bar{z}) a_i^T ) x <= c
           -(1/n \sum_{i=1}^n (z_i - \bar{z}) a_i^T ) x <= c
    
        by introducing slack variables and decomposing x as 
        x = x_plus - x_minus.

        Let B_tilde = (1/n \sum_{i=1}^n (z_i - \bar{z}) a_i^T ). Then the problem
        can be formulated as

        minimize f(w) 
        subject to

            [ B_tilde    -B_tilde    I_d     0  ]      [c]
             -B_tilde     B_tilde     0      I_d]  w = [c]
                                                   w >= 0,
    
        where w = [x_plus, x_minus, s1, s2].

        Parameters:
         A - data matrix, size n x d.
         B - the coefficient matrix, size 2r x 4d.
         b - rhs of equality constraints, size 2r.
         y - Labels for data points.
         r - Number of sensitive attributes.
         alpha_backtrack, beta_backtrack - line search
         parameters.
         sketching_size should be of order 0.01n or something like that. 
         Experiment with this parameter.
         sketching_strategy= 1 corresponds to no sketching, 2 corresponds to 
                          Gaussian sketch, 3 corresponds to SJLT.

    """
    def __init__(self, B, b, A, y, r, alpha_backtrack, beta_backtrack, 
                 sketching_size, sketching_strategy):
        super().__init__(B, b, alpha_backtrack, beta_backtrack) 
        self.n, self.original_d = A.shape                                  # Number of data points and number of original variables.
        self.A = A 
        self.sketching_size = sketching_size 
        self.sketching_strategy = sketching_strategy
        self.y = y                                                         # Must be a column vector with one dimension (not zero).
        self.r = r                                                         # Number of sensitive attributes
        self.yA = self.y * self.A 
    def evaluate_grad(self, w):
        x_plus = w[0:self.original_d]
        x_minus = w[self.original_d:2*self.original_d]
        #cached_quantity1_cmp = self.A.T * self.y.reshape(len(self.y), )
        cached_quantity1 = np.exp(self.yA @ (x_plus - x_minus))
        dh = np.divide(cached_quantity1, 1 + cached_quantity1)
        #grad_cmp = 1/self.n*np.vstack((cached_quantity1_cmp @ dh, -cached_quantity1_cmp @ dh,
        #                           np.zeros((self.r, 1)), np.zeros((self.r, 1))))
        cached_quantity2 = self.A.T @ (self.y * dh)
        grad = 1/self.n*np.vstack((cached_quantity2, -cached_quantity2,
                                   np.zeros((self.r, 1)), np.zeros((self.r, 1))))

        return grad 
    
    def evaluate_hessian(self, w):

        cached_quantity1 = self.y * (self.A @ (w[0:self.original_d] - w[self.original_d:2*self.original_d]))
        dsqrt = np.divide(np.exp(cached_quantity1/2), np.exp(cached_quantity1) + 1)
        #dsqrt = np.divide(np.exp(self.y * self.A @ (w[0:self.d] - w[self.d:2*self.d])/2),
        #                   np.exp(self.y * self.A @ (w[0:self.d] - w[self.d:2*self.d])) + 1)

        # No sketching.
        if self.sketching_strategy == 1:
            stored_quantity = dsqrt * (self.yA)

        # Gaussian sketch.
        #elif self.sketching_strategy == 2:
        #    stored_quantity = gaussian(torch.tensor(dsqrt * (self.y * self.A)), self.sketching_size)
        #    stored_quantity = stored_quantity.numpy()

        # Countsketch with numpy.
        elif self.sketching_strategy == 3: 
            #stored_quantity = sjlt(torch.tensor(dsqrt * (self.y * self.A)), self.sketching_size)                                    
            #stored_quantity = stored_quantity.numpy()

            # Generate sketch matrix.
            indices = np.vstack([np.random.choice(self.sketching_size, self.n).reshape((1,-1)), np.arange(self.n)])
            values = np.random.choice(np.array([-1,1], dtype=np.float64), size=self.n)
            S = csr_matrix((values, indices), (self.sketching_size, self.n))

            # Sketch the square root of the Hessian.
            stored_quantity = S @ (dsqrt * (self.y * self.A))
        
        # Countsketch with torch.
        #elif self.sketching_strategy == 4:
        #    indices = np.vstack([np.random.choice(self.sketching_size, self.n).reshape((1,-1)), np.arange(self.n)])
        #    values = np.random.choice(np.array([-1,1], dtype=np.float64), size=self.n)
            #S = torch.sparse_coo_tensor(indices, values, (self.sketching_size, self.n)).to(matrix.device)
        #    S = torch.sparse_coo_tensor(indices, values, (self.sketching_size, self.n))
        #    stored_quantity = S @ torch.tensor(dsqrt * (self.y * self.A))
        #    stored_quantity = stored_quantity.numpy()
        
        #SJLT 
        elif self.sketching_strategy == 5:
            mysketcher = SketchesDickens.RandomProjection((dsqrt * (self.y * self.A)), self.sketching_size, 'sjlt', 8)
        #   stored_quantity = mysketcher.sketch()

        cached_quantity2 = 1/self.n*stored_quantity.T @ stored_quantity

        #A_tilde = np.hstack((self.y * self.A, -self.y * self.A, np.zeros((self.n, 2*self.r))))
        #h_hess = 1/self.n*np.divide(np.exp(A_tilde @ w), np.square(1+np.exp(A_tilde @ w)))
        #cached_quantity2 = (self.A.T * self.y.reshape(len(self.y), )) @ np.diag(h_hess.reshape(len(h_hess), )) @ (self.y * self.A) 
        row = np.hstack((cached_quantity2, -cached_quantity2, np.zeros((self.original_d, self.r)), np.zeros((self.original_d, self.r))))
        self.hess = np.vstack((row, -row, np.zeros((2*self.r, 2*self.original_d+ 2*self.r))))
        
    def compute_obj(self, w):
        x_original = w[0:self.original_d] - w[self.original_d:2*self.original_d]
        obj = np.sum(np.log((np.exp(self.A @ x_original) + 1)))
        return obj