import numpy as np
import torch

def evaluate_performance(environment, policy, GAMMA, agent):

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
        
    mu = (1-GAMMA) * np.linalg.inv(np.eye(nS) - GAMMA * pi2 @ P).T @ mu0

    J = 1 / (1 - GAMMA) * mu @ pi2 @ r

    print(f"Performance J: {J}")

    value = agent.algorithm.critic(torch.tensor(np.arange(nS))).detach().numpy()
    values = [item for item in value for i in range(nA)]

    # Essendo calcolati in maniera approssimata i delta vanno usati per definire la nuova policy
    # che va poi normalizzata

    delta = values - GAMMA * P @ value
    pi2_star = pi2 * delta[None, :]
    pi2_star = pi2_star / np.sum(pi2_star, axis=1)[:, None]

    nu_star = (1 - GAMMA) * np.linalg.inv(np.eye(nS) - GAMMA * pi2_star @ P).T @ mu0

    J_star = 1 / (1 - GAMMA) * nu_star @ pi2_star @ r

    print(f"Performance J_star: {J_star}")