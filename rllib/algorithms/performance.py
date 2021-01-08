import numpy as np

def evaluate_performance(environment, policy, GAMMA):

    pi = policy.table.detach().numpy()

    nS = pi.shape[0]
    nA = pi.shape[1]

    mu0 = np.full(nS, 1/nS)

    pi2 = np.tile(pi, (1, nS)) * np.kron(np.eye(nS), np.ones((1,nA)))

    r = []

    for k in environment.transitions:
        r.append(environment.transitions[k][0]["reward"])
    
    r = np.array(r)

    P = []

    for k in environment.transitions:
        P_row = []
        for i in range(nS):
            P_row.append(environment.transitions[k][i]["probability"])
        P.append(P_row)
    
    P = np.array(P)
        
    mu = (1-GAMMA) * np.linalg.inv(np.eye(nS) - GAMMA * pi2 @ P).T @ mu0
    J = mu @ pi2 @ r

    print(f"Performance J: {J}")

    # nu_star = (mu @ pi2) * r / delta

    # J_star = nu_star @ r 

    # print(f"Performance J_star: {J_star}"")