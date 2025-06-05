import time

import numpy as np


def move_beetles(population, fitness, best_position, g_best, pop_size, lb, ub):
    new_population = np.copy(population)
    for i in range(pop_size):
        if np.random.rand() < 0.5:
            new_population[i] = population[i] + np.random.uniform(0, 1) * (best_position - population[i])
        else:
            new_population[i] = population[i] + np.random.uniform(0, 1) * (g_best - population[i])
    return np.clip(new_population, lb, ub)


# Dung Beetle Optimizer (DBO)
def DBO(population, fobj, VRmin, VRmax, max_iter):
    pop_size, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    fitness = fobj(population)
    best_idx = np.argmin(fitness)
    best_position = population[best_idx]
    g_best = best_position
    g_best_fitness = fitness[best_idx]

    convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    for t in range(max_iter):
        new_population = move_beetles(population, fitness, best_position, g_best, pop_size, lb, ub)
        new_fitness = fobj(new_population)

        for i in range(pop_size):
            if new_fitness[i] < fitness[i]:
                population[i] = new_population[i]
                fitness[i] = new_fitness[i]

        best_idx = np.argmin(fitness)
        best_position = population[best_idx]

        if fitness[best_idx] < g_best_fitness:
            g_best = best_position
            g_best_fitness = fitness[best_idx]

        convergence_curve[t] = g_best_fitness
        t = t + 1
    g_best_fitness = convergence_curve[max_iter - 1][0]
    ct = time.time() - ct

    return g_best_fitness, convergence_curve, g_best, ct
