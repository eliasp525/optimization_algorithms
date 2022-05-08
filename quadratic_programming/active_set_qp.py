import numpy as np
from eqp import EQP
import random


def active_set_QP(P : np.matrix, q : np.matrix, Aiq : np.matrix, biq : np.matrix, Aeq : np.matrix, beq : np.matrix, verbose = True):
    '''
    Solves problem on the form
    min { 1/2 x^T P x + q^T x }
    Aiq x >= biq
    Aeq x = beq
    ===========================
    Convex problem if P is positive semi-definite (P >= 0)
    Feasible set is a polyhedron
    ===========================
    Theorem 16.4:
    If x* satisfies KKT and P is positive semi-definite, 
    then x* is global solution.
    ===========================
    Active-set methods for QP differ from the simplex method in that the iterates
    (and the solution x*) are not necessarily vertices of the feasible region.
    '''
    if np.size(Aiq, 0) != np.size(biq, 0):
        raise ValueError("Sizes for inequality constraints does not match.")
    if np.size(Aeq, 0) != np.size(beq, 0):
        raise ValueError("Sizes for equality constraints does not match.")
    
    nx = np.size(P, 0)

    if np.size(q) == 0:
        q = np.zeros((nx, 1))
    
    inequalities = np.size(Aiq) > 0
    equailites = np.size(Aeq) > 0
    iq_constraints, eq_constraints = 0, 0
    if inequalities:
        iq_constraints = np.size(Aiq, 0)
    if equailites:
        eq_constraints = np.size(Aeq, 0)
    
    if equailites and not inequalities:
        return EQP(P, q, Aeq, beq)
    
    if inequalities:
        '''
        Algorithm 16.3 in Nocedal
        ===========================
        '''
        # just putting every constraint into Acon and bcon
        if equailites:
            Acon = np.block([[Aiq],[Aeq]])
            bcon = np.block([[biq], [beq]])
        else:
            Acon = Aiq
            bcon = biq
        
        if verbose:
            print(f"Acon:\n{Acon}")
            print(f"bcon:\n{bcon}")

        #initializing variables
        working_set = set([2, 4])
        inequality_set = set(np.linspace(0, iq_constraints-1, iq_constraints, dtype=int))
        equaility_set = set(np.linspace(iq_constraints, iq_constraints+eq_constraints-1, eq_constraints, dtype=int))
        if verbose:
            print("inequality_set:\n", inequality_set, "\nequality_set:\n", equaility_set)
        
        X = list() # currently not returned, but could be become useful
        xk = np.matrix([[2], [0]])  # np.zeros((nx, 1))

        DECIMALS = 5
        iterations = 0
        while True:
            iterations += 1
            xk = xk.round(DECIMALS)
            X.append(xk)
            
            if verbose:
                print(f"=============== Iteration {iterations} ===============")
                print(f"Current x:\n{xk}")
                print(f"Current working set: {working_set}")
            
            #define EQP based on working set
            gk = np.matmul(P, xk) + q
            Aeqk = np.matrix([np.array(Aiq[i, :])[0] for i in working_set])
            beqk = np.zeros((np.size(Aeqk, 0), 1))
            if verbose:
                print(f"Aeqk:\n{Aeqk}")
                print(f"beqk:\n{beqk}")
            
            pk, lmda = EQP(P, gk, Aeqk, beqk)
            pk = pk.round(DECIMALS)
            
            if verbose:
                print(f"Got pk\n:{pk}\n and lmdas:\n{lmda}")
            
            if np.all(pk == 0):
                '''
                When p == 0 we reach the point x̂ 
                that minimizes the quadratic objective 
                function over its current working set.
                '''
                # Compute Lagrange multipliers λ̂i that satisfy (16.42) based on current working set
                # (these are already calculated in lmda = EQP(...))
                if np.all(lmda >= 0):
                    if verbose:
                        print(f"Found solution:\n{xk}")
                    return xk, lmda
                else:
                    iq_in_work = list(working_set.intersection(inequality_set))
                    if iq_in_work:
                        j = iq_in_work[np.argmin(lmda)]
                        # x_{k+1} = x_{k} # not necessary to do, only here for context
                        working_set.remove(j)
                        if verbose:
                            print(f"Removed {j} from working set")
            else:
                nowork_set = inequality_set.union(equaility_set) - working_set
                upper_limits_alphak = list()
                for _, i in enumerate(nowork_set):
                    aiT = Acon[i, :]
                    aiTpk = np.matmul(aiT, pk)
                    if not np.all(aiTpk < 0):
                        continue
                    bi = bcon[i, :]
                    aiTxk = np.matmul(aiT, xk)
                    limit = float((bi-aiTxk)/aiTpk)
                    upper_limits_alphak.append({"limit": limit, "i": i})
                
                alphak = min({"limit": 1}, min(upper_limits_alphak, key=lambda elem: elem["limit"]), key=lambda elem: elem["limit"])
                if verbose:
                    print(f"alphak:\n{alphak['limit']}")
                
                # ensure type int or float
                if type(alphak) == np.matrix:
                    if np.size(alphak) != 1:
                        raise Exception
                    alphak = alphak[0, 0]
                xk = xk + alphak["limit"]*pk
                if alphak["limit"] != 1:
                    ''' 
                    if there are blocking constraints, 
                    add one of the blocking constraints
                    '''
                    working_set.add(alphak["i"])
                    if verbose:
                        print(f"Added {alphak['i']} to working set")
                else:
                    '''W_{k+1} = W_k not necessary to do, only here for context'''
                    pass



# Example 16.2 from Nocedal Numerical Optimization 2nd Edition
# expected xsol = [[2],[-1],[1]], lmdas = [[3],[-2]] 
# only EQP
xsol, lmdas = active_set_QP(
    np.matrix([[6, 2, 1], [2, 5, 2], [1, 2, 4]]),
    np.matrix([[-8], [-3], [-3]]),
    np.matrix([]),
    np.matrix([]),
    np.matrix([[1, 0, 1], [0, 1, 1]]),
    np.matrix([[3], [0]])
)
print(f"=== Solution ===\nx:\n{xsol}\nmultipliers:\n{lmdas}")

# Example 16.4 from Nocedal Numerical Optimization 2nd Edition
# expected xsol = [[1.4], [1.7]], lmdas = [[0.8]]
xsol, lmdas = active_set_QP(
    np.matrix([[2, 0], [0, 2]]),
    np.matrix([[-2], [-5]]),
    np.matrix([[1, -2],[-1, -2], [-1, 2], [1, 0], [0, 1]]),
    np.matrix([[-2], [-6], [-2], [0], [0]]),
    np.matrix([]),
    np.matrix([]),
    verbose=False
)
print(f"=== Solution ===\nx:\n{xsol}\nmultipliers:\n{lmdas}")