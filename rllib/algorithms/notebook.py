from cvxopt import solvers, matrix, spdiag, mul, div
import numpy as np


def run_notebook(environment, policy, GAMMA, agent):
    
    pi = policy.table.detach().numpy().T
    pi = np.exp(pi) / np.sum(np.exp(pi), axis=1)[:, None]

    nS, nA = pi.shape

    mu0 = np.full(nS, 1/nS)

    pi2 = np.tile(pi, (1, nS)) * np.kron(np.eye(nS), np.ones((1, nA)))

    r = []

    for k in sorted(environment.transitions.keys(), key=lambda x: x[0] * nS + x[1]):
        r.append(environment.transitions[k][0]["reward"])
    
    r = np.array(r)

    P = []

    for k in sorted(environment.transitions.keys(), key=lambda x: x[0] * nS + x[1]):
        P_row = []
        for i in range(nS):
            P_row.append(environment.transitions[k][i]["probability"])
        P.append(P_row)
    
    P = np.array(P)

    mu = (1 - GAMMA) * np.linalg.inv(np.eye(nS) - GAMMA * pi2 @ P).T @ mu0

    A = np.kron(np.eye(nS), np.ones((1, nA))) - GAMMA * P.T

    mu__ = (mu @ pi2)[:, None, None]
    r__ = r[:, None, None]
    P1__ = P[:, :, None]
    P2__ = P[:, None, :]
    temp__ = np.kron(np.eye(nS), np.ones((nA, 1)))
    Ind1__ = temp__[:, :, None]
    Ind2__ = temp__[:, None, :]
    _lambda__ = np.ones((nS, 1))
    Alambda__ = (A.T @ _lambda__)[:, None]

    def objective(_lambda):
        lambda__ = np.array(_lambda)
        Alambda__ = (A.T @ lambda__)[:, None]
        objective__ = np.sum(mu__ * r__ * Alambda__ ** .5) - (1 - GAMMA) * mu0 @ lambda__ 
        return - objective__

    def gradient(_lambda):
        lambda__ = np.array(_lambda)
        Alambda__ = (A.T @ lambda__)[:, None]
        gradient__ = np.sum(mu__ * r__ * Alambda__ ** (- .5) * (Ind1__ - GAMMA * P1__), axis=0).squeeze() - (1 - GAMMA) * mu0
        return - matrix(gradient__).T

    def hessian(_lambda):
        lambda__ = np.array(_lambda)
        Alambda__ = (A.T @ lambda__)[:, None]
        hessian__ = - .5 * np.sum(mu__ * r__ * Alambda__ ** (- 1.5) * (Ind1__ - GAMMA * P1__) * (Ind2__ - GAMMA * P2__), axis=0)
        return - matrix(hessian__)

    _lambda0 = matrix(np.ones((nS, 1)))

    #Objective function
    def F(x=None, z=None):
        
        if x is None and z is None:
            return 0, _lambda0
        if z is None:
            lambda__ = np.array(x)
            Alambda__ = (A.T @ lambda__)[:, None]            
            return objective(x), gradient(x)
        return objective(x), gradient(x), z[0] * hessian(x)

    res = solvers.cp(F)

    lambda_star = np.array(res['x']).ravel()
    nu_star_dual = (mu @ pi2) * r / (A.T @ lambda_star) ** .5
    J_star = 1 / (1 - GAMMA) * nu_star_dual @ r
    print(f"Performance J_star from optimization: {J_star}")