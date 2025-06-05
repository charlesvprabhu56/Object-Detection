import random as rn
import time
from math import floor
import numpy as np


# Rat Swarm Optimizer (RSO)
def RSO(Positions, fobj, VRmin, VRmax, Max_iterations):
    N, dim = Positions.shape[0], Positions.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    Score = float('inf')
    Position = np.zeros((dim, 1))
    Convergence_curve = np.zeros((Max_iterations, 1))

    l = 0
    x = 1
    y = 5
    R = floor((y - x) * rn.random() + x)
    t = 0
    ct = time.time()
    while t < Max_iterations:
        for i in range(N, 1):
            Flag4Upper_bound = Positions[i, :] > ub
            Flag4Lower_bound = Positions[i, :] < lb
            Positions[i, :] = (Positions[i, :] * (
                ~(Flag4Upper_bound + Flag4Lower_bound))) + ub * Flag4Upper_bound + lb * Flag4Lower_bound

            fitness = fobj(Positions[i, :])

            if fitness < Score:
                Score = fitness
                Position = Positions[i, :]

        A = R - l * ((R) / Max_iterations)

        for i in range(N):
            for j in range(dim):
                C = 2 * rn.random()
                P_vec = A * Positions[i, j] + abs(C * ((Position - Positions[i, j])))
                P_final = Position - P_vec
                Positions[i, j] = P_final[j]

        Convergence_curve[t] = Score
        t = t + 1
    Score = Convergence_curve[Max_iterations - 1][0]
    ct = time.time() - ct
    return Score, Convergence_curve, Position, ct
