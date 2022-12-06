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
    # OBS: This line search is not for a general problem. It has been modified 
    # to the special case of logistic regression with ell-1 regularization.
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
        m, d = self.B.shape                              # d is the dimension of the optimization variable 
        x, _lambda, nu = x0, _lambda0, nu0               # (which is not the same as the dimension of the feature vector)
         

        # Define parameters for tracking the progress.
        times, residual_norms, residual_opt_conds = [], [], []
        alphas, rd = np.zeros((max_iter, )), np.zeros((max_iter, ))
        rcent, rfeas = np.zeros((max_iter, )), np.zeros((max_iter, ))
        tic = time()

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

            ###############################################################################
            # Apply one correction step. The extra cost is one Hessian-vector product.
            # In this naive implementation we compute the Hessian-vector product 
            # by actually forming the true Hessian... The point is to see if 
            # adding a correction step helps in terms of the achieved accuracy.
            if self.sketching_strategy == 3 and self.correction:
                print("Applying correction step.")
                sketched_hessian = self.hess                                         # Store the sketched Hessian 
                self.sketching_strategy = 1              
                self.evaluate_hessian(x)                                             # Evaluate the true Hessian
                self.sketching_strategy = 3                                          
                correction_term = (self.hess - sketched_hessian)@dx               

                new_rhs = residuals
                new_rhs[0:d] = new_rhs[0:d] + correction_term

                # We could factor coeff_matrix earlier and reuse the factorization,
                # but profiling shows that the linear system solve is (for our problems)
                # very cheap compared to forming the sketched Hessian.
                dir = -LA.solve(coeff_matrix, residuals)
                dx = dir[0:d]
                dlambda = dir[d:2*d]
                dnu = dir[2*d:]


            ################################################################################
            alpha = self.backtracking(x, _lambda, nu, dx, dlambda, dnu, t, LA.norm(residuals))      
            alphas[iter] = alpha

            x = x + alpha*dx
            _lambda = _lambda + alpha*dlambda 
            nu = nu + alpha*dnu 

        obj = self.compute_obj(x)
        return x, _lambda, nu, residual_norms, times, alphas, obj, rd, rcent, rfeas, residual_opt_conds




class log_reg_ell1_solver(Solver):
    """ A value, gradient and Hessian oracle for the objective function 
        when logistic regression with ell1-regularization is formulated on 
        the form

            min          f(x)
            subject to   Bx = b,
                         x >= 0.

        
        B, b - constraint data
        gamma - regularization parameter
        A, y - data matrix and labels
        alpha_backtrack, beta_backtrack - parameters for backtracking
        
        sketching_strategy - either 1 or 3. 1 = no sketch. 3 = countsketch.
        correction - True if one correction step should be used when computing
                     the sketched search direction.

    """
    def __init__(self, B, b, gamma, A, y, alpha_backtrack, beta_backtrack,
                 sketching_size, sketching_strategy, correction):
        super().__init__(B, b, alpha_backtrack, beta_backtrack)                                                       
        self.n, self.original_d = A.shape                                                # Number of data points and the original dimension of the feature vector.
        self.A = A                                                                       
        self.sketching_size = sketching_size  
        self.sketching_strategy = sketching_strategy
        self.y = y                                                                       # Must be a column vector with one dimension (not zero).
        self.yA = self.y * self.A                                                        # The data matrix with scaled columns. Just for simplicity.
        self.gamma = gamma                                                               
        self.correction = correction

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
 






""" Builds a problem of logistic regression with ell-1 regularization.
    The data comes from a Gaussian distribution.
    n = number of data points
    d = dimension of feature vector
    rho = parameter used to build the covariance matrix
"""
def build_log_reg_ell1_problem(n, d, rho):


    # Generate labels and data matrix.
    y = np.zeros((n, 1))
    for i in range(0, n):
        y[i] = 1 if np.random.random() < 0.5 else -1

    # Generate data matrix A.
    cov_matrix = np.zeros((d, d))
    for i in range(0, d):
        for j in range(0, d):
            cov_matrix[i, j] = 2*(rho**np.abs(i-j))

    A = np.random.multivariate_normal(np.zeros((d, )), cov_matrix, size=n)


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
