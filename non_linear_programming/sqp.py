## Nonlinear optimization problems
'''
min f(x)
 x
s.t.
c_i(x) = 0, i in equalities
c_i(x) >= 0, i in inequalities
'''

def ESQP():
    '''
    SQP for equality-constrained nonlinear programming problems
    Algorithm 18.1 in Nocedal
    '''
    pass

def SQP():
    '''
    SQP for general nonlinear programming problems
    =======================
    SQP needs Hessian of Lagrangian, but this require second derivatives of objective and constraints,
    which may be expensive
    Quasi-Newton (BFGS) very successful for unconstrained optimization – can we do the same in the
    constrained case? Yes
    '''
    pass


## Merit function ##
'''
Measure progress in both objective and constraints
'''

