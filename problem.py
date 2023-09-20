from __future__ import division
import numpy as np
# git clone git@github.com:antonior92/ip-nonlinear-solver.git
from ipsolver import minimize_constrained, NonlinearConstraint, BoxConstraint
from ipsolver._large_scale_constrained.inexect_interior_point import inexect_interior_point

# Define objective function and derivatives
fun = lambda x: 1/2*(x[0] - 2) + 1/2*(x[1] - 1/2)
grad = lambda x: np.array([x[0] - 2, x[1] - 1/2])
hess =  lambda x: np.eye(2)
# Define nonlinear constraint
c = lambda x: np.array([1/(x[0] + 1) - x[1],])
c_jac = lambda x: np.array([[-1/(x[0] + 1)**2, -1]])
c_hess = lambda x, v: 2*v[0]*np.array([[1/(x[0] + 1)**3, 0], [0, 0]])
d_jac = lambda x: 2*np.array([[1/(x[0] + 1)**3, 0], [0,0]])
nonlinear = NonlinearConstraint(c, ('greater', 1/4), c_jac, c_hess)
# Define box constraint
box = BoxConstraint(("greater",))

# Define initial point
x0 = np.array([0, 0])
# Apply solver
result = minimize_constrained(fun, x0, grad, hess, (nonlinear, box))

# Print result
print(result.x)

print(inexect_interior_point(np.array([1/2,1/2]), c, 1, c_jac, c_hess, 0.1, gamma=0.75, x0=None))

