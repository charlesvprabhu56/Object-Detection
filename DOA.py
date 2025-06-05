import random
import time

import numpy as np


def select_parents(population, fitness, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(list(zip(population, fitness)), tournament_size)
        selected.append(min(tournament, key=lambda x: x[1])[0])  # Choose the best in the tournament
    return selected


def crossover(parent1, parent2, crossover_rate=0.7):
    if random.random() < crossover_rate:
        alpha = random.random()  # Blend ratio
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = alpha * parent2 + (1 - alpha) * parent1
        return child1, child2
    return parent1, parent2


def mutate(doll, mutation_rate=0.1, mutation_strength=0.5):
    if random.random() < mutation_rate:
        mutation = np.random.normal(0, mutation_strength, len(doll))
        doll = doll + mutation
    return doll


# Dollmaker Optimization Algorithm (DOA)
def DOA(population, objective_function, VRmin, VRmax, generations):
    pop_size, dim = population.shape[0], population.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    best_solution = np.zeros((dim, 1))
    best_fitness = float('inf')
    Convergence_curve = np.zeros((generations, 1))

    t = 0
    ct = time.time()
    for gen in range(generations):
        # Step 2: Evaluate fitness
        fitness = objective_function(population)

        # Step 3: Selection (Tournament)
        parents = select_parents(population, fitness)

        # Step 4: Crossover and Mutation
        next_population = []
        for i in range(0, len(parents), 2):
            parent1, parent2 = parents[i], parents[min(i + 1, len(parents) - 1)]
            child1, child2 = crossover(np.array(parent1), np.array(parent2))
            next_population.append(mutate(child1))
            next_population.append(mutate(child2))

        # Step 5: Create new population
        population = next_population[:pop_size]

        # Log the best solution of the generation
        best_fitness = min(fitness)
        best_solution = population[fitness.index(best_fitness)]
    Convergence_curve[t] = best_fitness
    t = t + 1
    best_fitness = Convergence_curve[generations - 1][0]
    ct = time.time() - ct
    # Return the best solution found
    return best_fitness, Convergence_curve, best_solution, ct


