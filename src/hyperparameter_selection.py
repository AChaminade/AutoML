# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:46:25 2024

@author: adrgc
"""

# Librerías
#-----------------------------------------------------------------------------#
# Librerías de tratamiento de datos
import numpy as np
import random

# Librerías de entorno
import tqdm

# Librerías de ML
from sklearn.model_selection import cross_val_score

#=============================================================================#

# Función para crear un individuo
def create_individual(hyperparameters):
    return {k: random.choice(v) for k, v in hyperparameters.items()}

# Función para evaluar un individuo
def evaluate_individual(individual, model, X, y):
    model.set_params(**individual)
    scores = cross_val_score(model, X, y, cv = 3)
    return np.mean(scores)

# Función para cruzar dos individuos
def crossover(individual1, individual2):
    child = individual1.copy()
    for param in individual2:
        if random.random() > 0.5:
            child[param] = individual2[param]
    return child

# Función para mutar un individuo
def mutate(individual, hyperparameters):
    mutation_param = random.choice(list(hyperparameters.keys()))
    individual[mutation_param] = random.choice(hyperparameters[mutation_param])
    return individual

# Función para seleccionar la siguiente generación
def select_population_tournament(
        population, 
        fitness, 
        num_parents, 
        tournament_size = 3
        ):
    selected_parents = []
    for _ in range(num_parents):

        tournament_indices = np.random.choice(
            len(population), 
            tournament_size,
            replace = False
            
            )
        tournament_individuals = [population[i] for i in tournament_indices]
        tournament_fitness = [fitness[i] for i in tournament_indices]

        best_in_tournament_idx = np.argmax(tournament_fitness)
        selected_parents.append(tournament_individuals[best_in_tournament_idx])

    return selected_parents

# Algoritmo genético
def genetic_algorithm_tournament(
        hyperparameters, 
        model, 
        X, 
        y, 
        population_size = 20, 
        num_parents = 10, 
        generations = 5, 
        mutation_rate = 0.2, 
        tournament_size = 3
        ):
    population = [
        create_individual(hyperparameters) for _ in range(population_size)
        ]
    
    for _ in tqdm.tqdm(range(generations)):
        fitness = [
            evaluate_individual(individual, model, X, y) for 
            individual in population
            ]
        parents = select_population_tournament(
            population, 
            fitness,
            num_parents, 
            tournament_size
            )

        next_population = parents.copy()
        while len(next_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            if random.random() < mutation_rate:
                child = mutate(child, hyperparameters)
            next_population.append(child)

        population = next_population

    best_fitness = [
        evaluate_individual(individual, model, X, y) for 
        individual in population
        ]
    best_index = np.argmax(best_fitness)
    return population[best_index]