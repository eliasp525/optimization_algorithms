import numpy as np
import scipy
import time

def EQP(P : np.matrix, q : np.matrix, Aeq : np.matrix, beq : np.matrix, method = "fullspace", verbose=False):
    '''
    Solves problem on the form
    min { 1/2 x^T P x + q^T x }
    Aeq x = beq
    ===========================
    Convex problem if P is positive semi-definite (P >= 0)
    Feasible set is a polyhedron
    '''
    available_methods = ["fullspace", "reducedspace"]
    if method not in available_methods:
        raise Exception("Method %s not available. Available methods are %s" % (method, available_methods))
    
    nx = np.size(P, 0)

    if np.size(q) == 0:
        q = np.zeros((nx, 1))
    
    if np.size(Aeq) == 0:
        return np.linalg.solve(P, -q), np.matrix([[]])
    
    '''
    FULL SPACE
    ===================================
    Problem can be converted to linear system:
        H       *  r      =    w  | Alternatively with x* = x + p:
    [P  -Aeq^T]   [x*   ]    [-q] | [P  -Aeq^T]   [p    ]    [-q-Px  ]  (   [-g])
    [Aeq   0  ] * [lmda*] =  [ b] | [Aeq   0  ] * [lmda*] =  [ b-Aeqx]  ( = [-h])
    ===================================
    If A full row rank and Z^T P Z > 0, then H is non-singular 
    and we are guarranteed a unique solution to the EQP. (Lemma 16.1)
    Z : columns in Z span null(Aeq) <=> columns are basis for null(Aeq)
    '''
    if method == "fullspace":
        tinit = time.time()
        H = np.block([[P, -np.transpose(Aeq)],
                        [Aeq, np.zeros((np.size(Aeq, 0), np.size(Aeq, 0)))]])
        w = np.block([[ -q], 
                        [beq]])
        r = np.linalg.solve(H, w)
        tfinish = time.time()-tinit
        if verbose:
            print(f"Found full space solution r:\n{r}")
            print(f"Time used for full space: {tfinish*1000:.4f} ms")
            print("=============================================")
        x_sol = r[:nx]
        lmda = r[nx:]
        return x_sol, lmda
    
    '''
    REDUCED SPACE (Efficient if n-m ≪ n)
    ====================================
    Solve two much smaller systems using LU and Cholesky (both with complexity that scales with n^3)
    Main complexity is calculating basis for nullspace. Usual method is using QR.
    ====================================
    Alternative to direct methods: Iterative methods,
    for very large systems these can be parallelized
    ====================================
    (remember x* = x + p)
    Problem converted to:
    I   : (Aeq Y) p_Y = beq - Aeq x
    II  : (z^T P z)p_Z = -Z^T P Y p_Y - Z^T (q + P x)
    III : p = Y p_Y + Z p_Z
    where
    Z   : Basis for null(Aeq)
    Y   : Basis for range(Aeq^T)
    ====================================
    To find p:
    1. Find Y, Z s.t. p = Y p_Y + Z p_Z
    2. Solve (A Y) p_Y = -h, 
    for p_Y (LU decomposition), -h := (beq - Aeq x)
    3. Solve (Z^T P Z) p_Z = -Z^T P Y p_Y - Z^T g
    (with cholesky)
    4. p = Y p_Y + Z p_Z 
    '''

    if method == "reducedspace":
        tinit = time.time()
        Z = scipy.linalg.null_space(Aeq)
        ZT = np.transpose(Z)
        Y = scipy.linalg.orth(np.transpose(Aeq))
        # could be smart to choose Y such that: Aeq Y = I.
        
        x = np.zeros((nx, 1)) # Suppose we have x = [0, 0, ..., 0]^T
        h = -beq            # -h = beq - (Aeq*x = 0)
        g = q               # g = c + (P x = 0)

        p_Y = np.linalg.solve(Aeq*Y, -h)
        p_Z = np.linalg.solve(ZT*P*Z, -ZT*P*Y*p_Y - np.matmul(ZT, g))
        p = Y*p_Y + Z*p_Z   # x* = (x=0) + p = p
        # finding lagrange multipliers from first block of KKT system
        # (Aeq Y)^T lmda* = Y^T (g + Pp)
        lmda = np.linalg.solve(np.transpose(Aeq*Y), np.transpose(Y)*(g + P*p))
        tfinish = time.time()-tinit
        if verbose:
            print(f"Found reduced space solution:\n{p}")
            print(f"Lagrange multipliers:\n{lmda}")
            print(f"Time used for reduced space: {tfinish*1000:.4f} ms")
            print("=============================================")
        return p, lmda
    # THIS IS NOT DONE THE OPTIMAL WAY AS TIME USAGE CLEARLY SHOWS
    # ONLY FOR PRACTICE



# expected: r = [[0.5], [0.5], [0.5]]
EQP(np.matrix([[1, 0], [0, 1]]),
        np.matrix([]),
        np.matrix([[1, 1]]),
        np.matrix([[1]]))

# Example 16.2 from Nocedal Numerical Optimization 2nd Edition
# expected r = [[2],[-1],[1],[3],[-2]]
EQP(np.matrix([[6, 2, 1], [2, 5, 2], [1, 2, 4]]),
        np.matrix([[-8], [-3], [-3]]),
        np.matrix([[1, 0, 1], [0, 1, 1]]),
        np.matrix([[3], [0]]))