"""Inexect interior point method.

References
----------
.. [1]  Miculescu, David, and Sertac Karaman.
        "Solving Large-scale Quadratic Programs with Tensor-train-based
         Algorithms with Application to Trajectory Optimization." 2018
         Annual American Control Conference (ACC). IEEE, 2018.
"""
import numpy as np
import scipy


def inexect_interior_point(g, c, m, grad_c, sec_grad_c, tol, gamma=0.75, x0=None):
    """Solve equality-constrained quadratic programming (EQP) problem.

        Solve ``min 1/2 x.T H x + x.t g``  subject to ``A x + b = 0, x.T Q x + k = 0``
        or ``c(x) = 0, x > 0``
        using direct factorization of the KKT system.

        Parameters
        ----------
        g : array_like, shape (n,)
            Linear part of objective function.
        c(x) : n to m  constraint function(m,)
        Twice continuously differentiable function.
        grad_c(x): n to m function, shape (m,)
            First derivative of c(x).
        sec_grad_c(x): n to m function, shape (m,)
            Second derivative of c(x).
        gamma : step size parameter.
        tol: break point



        Returns
        -------
        x : array_like, shape (n,)
            Solution of the KKT problem.

        """
    n = len(g)

    if x0:
        x = x0
    else:
        x = np.ones(n)


    lambda_, z, mu = np.ones(m), np.ones(n), 0

    # Define Hess
    one_ = np.ones(n)
    def H(d_x, d_lambda, d_z):
        h1 = g + np.dot(grad_c(d_x).T, np.atleast_2d(d_lambda)).reshape(-1,) - d_z
        h2 = c(d_x)
        h3 = np.dot(np.diag(d_x), np.diag(d_z)).dot(one_) - mu*one_

        return scipy.sparse.csr_matrix(np.hstack([h1, h2, h3]))

    def dif_H(d_x, d_lambda, d_z):
        h1 = np.r_['-1', sec_grad_c(d_x, d_lambda).T, grad_c(d_x).T, -np.eye(n)]
        h2 = np.r_['-1', grad_c(d_x), np.zeros((m,m)), np.zeros((m,n))]
        h3 = np.r_['-1', np.diag(d_z), np.zeros((n,m)), np.diag(d_x)]

        return scipy.sparse.csr_matrix(np.vstack([h1, h2, h3]))

    tau1 = min(np.dot(np.diag(z), np.diag(x)).dot(one_)) / np.dot(z, x) * m
    tau2 = np.dot(z, x) / norm(H(x, lambda_, z)[-n:])
    k = 0

    while True:
        # Choose σk, compute μk := σk*zk.T*sk / m, and choose ηˆk.
        # parameters θ and β are in (0, 1)
        sigma = 0.73
        teta = 0.98
        beta = 0.98
        mu = sigma * np.dot(z, x)/m
        eta_hat = 0.8

        # Approximate Newton direction ∆pk
        delta_p = scipy.sparse.linalg.spsolve(dif_H(x, lambda_, z), np.array(- H(x, lambda_, z)
                                              + np.r_[np.zeros(n),np.zeros(m),mu*one_]).reshape(-1,))

        # delta_p = AMEN(dif_H(x, lambda_, z), - H(x, lambda_, z) + mu*one_, norm(eta_hat*np.dot(z, x) / nc))

        d_x = delta_p[:len(g)]
        d_lambda = delta_p[len(g):m+len(g)]
        d_z = delta_p[m+len(g):]

        # Find step size alpha
        eta = sigma + eta_hat

        # Formula (5)
        f1 = lambda alpha_: min(np.dot(np.diag(z + alpha_*d_z), np.diag(x + alpha_*d_x)).dot(one_)) - \
                            tau1 * gamma * np.dot(z + alpha_*d_z, x + alpha_*d_x) / m

        f2 = lambda alpha_: np.dot(z, x) - tau2 * gamma *\
                            norm(H(x + alpha_*d_x, lambda_ + alpha_*d_lambda, z + alpha_ * d_z)[-n:])

        # Find (6)
        if f1(1) > 0:
            alpha1 = 1
        else:
            sol1 = scipy.optimize.root_scalar(f1, bracket=[0, 1])
            alpha1 = sol1.root
        if f2(1) > 0:
            alpha2 = 1
        else:
            sol2 = scipy.optimize.root_scalar(f2, bracket=[0, 1])
            alpha2 = sol2.root

        # Step 6
        alpha = min(alpha1, alpha2)
        H_module = norm(H(x, lambda_, z))
        while norm(H(x + alpha*d_x, lambda_ + alpha * d_lambda, z + alpha * d_z)) > (1-beta*(1-eta))*H_module:
            alpha = teta * alpha
            eta = 1 - teta * (1 - eta)
            print(alpha, eta, teta, beta)

        # Update solution
        x, lambda_, z = x + alpha*d_x, lambda_ + alpha * d_lambda, z + alpha * d_z

        # Stop criteria
        if norm(H(x, lambda_, z)) <= tol:
            break

        k += 1
    return x

def norm(x):
    return scipy.sparse.linalg.norm(x)
