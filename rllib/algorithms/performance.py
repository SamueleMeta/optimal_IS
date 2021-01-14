import numpy as np
import torch

def evaluate_performance(environment, policy, GAMMA, agent):

    pi = policy.table.detach().numpy()
    pi /= np.sum(pi, axis=1)[:,None]

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

    value = agent.algorithm.critic(torch.tensor(np.arange(nS))).detach().numpy()
    values = [item for item in value for i in range(nA)]

    delta = values - GAMMA * P @ value
 
    nu_star = (mu @ pi2) * r / delta

    J_star = nu_star @ r 

    print(f"Performance J_star: {J_star}")