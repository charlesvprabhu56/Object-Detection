import time

import numpy as np



# Update the position of a mongoose
def update_position(mongoose, best_mongoose, alpha, lb, ub):
    new_position = mongoose + alpha * np.random.uniform(-1, 1, mongoose.shape) * (best_mongoose - mongoose)
    # Ensure the new position is within bounds
    new_position = np.clip(new_position, lb, ub)
    return new_position


# Dwarf Mongoose Optimization (DMO) algorithm
def DMO(population, objective_function, VRmin, VRmax, max_iter):
    pop_size, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]
    fitness = objective_function(population[:])

    # Identify the best mongoose (initially)
    best_idx = np.argmin(fitness)
    best_mongoose = population[best_idx].copy()
    best_fitness = fitness[best_idx]
    Convergence_curve = np.zeros((max_iter, 1))

    t = 0
    ct = time.time()
    # Main loop of the DMO algorithm
    for t in range(max_iter):
        alpha = 2 * (1 - t / max_iter)  # Decrease linearly over iterations
        for i in range(pop_size):
            if i != best_idx:
                population[i] = update_position(population[i], best_mongoose, alpha, lb, ub)
                fitness[i] = objective_function(population[i])

        # Update the best mongoose
        new_best_idx = np.argmin(fitness)
        new_best_fitness = fitness[new_best_idx]
        if new_best_fitness < best_fitness:
            best_mongoose = population[new_best_idx].copy()
            best_fitness = new_best_fitness

    Convergence_curve[t] = best_fitness
    t = t + 1
    best_fitness = Convergence_curve[max_iter - 1][0]
    ct = time.time() - ct

    return best_fitness, Convergence_curve, best_mongoose, ct



