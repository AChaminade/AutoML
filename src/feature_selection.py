# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 12:53:59 2024

@author: adrgc
"""

# Librerías
#-----------------------------------------------------------------------------#
# Librerías de tratamiento de datos
import numpy as np

# Scikit-learn
from sklearn.metrics import (roc_auc_score, roc_curve, auc, 
                             f1_score, mean_squared_error)

# XGBoost
from lightgbm import LGBMClassifier, LGBMRegressor

# Librerías de entorno
import tqdm
from IPython.display import clear_output

# Librerías de visualización
import matplotlib.pyplot as plt

#=============================================================================#

def fitness_function(
        individual, 
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        task_type, 
        num_classes,
        random_state = 42
        ):
    selected_features_indices = np.where(individual == 1)[0]
    if len(selected_features_indices) == 0:
        return -np.inf if task_type == 'regression' else 0
    
    selected_features = [X_train.columns[i] for i in selected_features_indices]
    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]

    if task_type == 'classification':
        if num_classes == 2:
            model = LGBMClassifier(random_state = random_state, verbose = -1)
            model.fit(X_train_selected, y_train)
            predictions = model.predict_proba(X_test_selected)[:, 1]
            return roc_auc_score(y_test, predictions)
        else:
            model = LGBMClassifier(random_state = random_state, verbose = -1)
            model.fit(X_train_selected, y_train)
            predictions = model.predict(X_test_selected)
            return f1_score(y_test, predictions, average = 'macro')
    
    elif task_type == 'regression':
        model = LGBMRegressor(random_state = random_state, verbose = -1)
        model.fit(X_train_selected, y_train)
        predictions = model.predict(X_test_selected)
        return -mean_squared_error(y_test, predictions)

def initialize_population(population_size, num_features):
    return np.random.randint(2, size = (population_size, num_features))

def crossover(parent_1, parent_2):
    crossover_point = np.random.randint(1, len(parent_1))
    child_1 = np.concatenate([parent_1[:crossover_point], parent_2[crossover_point:]])
    child_2 = np.concatenate([parent_2[:crossover_point], parent_1[crossover_point:]])
    return child_1, child_2

def mutate(individual):
    mutation_index = np.random.randint(len(individual))
    individual[mutation_index] = 1 - individual[mutation_index]
    return individual

def tournament_selection(population, fitness_scores, tournament_size):
    selected_indices = np.random.choice(
        len(population), 
        tournament_size, 
        replace = False
        )
    tournament_fitness_scores = [fitness_scores[i] for i in selected_indices]
    winner_index = np.argmax(tournament_fitness_scores)
    return population[selected_indices[winner_index]]

def genetic_feature_selection(
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        population_size, 
        num_generations, 
        tournament_size, 
        task_type, 
        n_classes,
        random_state = 42
        ):
    num_features = X_train.shape[1]
    population = initialize_population(population_size, num_features)
    best_individual = None
    best_fitness = -np.inf if task_type == 'regression' else 0

    fitness_history = []
    best_fitness_history = []

    for generation in tqdm.tqdm(range(num_generations)):
        fitness_scores = [
            fitness_function(
                individual, 
                X_train, 
                X_test, 
                y_train, 
                y_test, 
                task_type, 
                n_classes,
                random_state
                ) for individual in population
        ]
        new_population = []

        current_best = np.argmax(fitness_scores)
        current_best_fitness = fitness_scores[current_best]
        average_fitness = np.mean(fitness_scores)

        fitness_history.append(average_fitness)
        best_fitness_history.append(current_best_fitness)

        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_individual = population[current_best]
        
        new_population.append(best_individual)

        for _ in range((population_size // 2) - 1):
            parent_1 = tournament_selection(population, fitness_scores, tournament_size)
            parent_2 = tournament_selection(population, fitness_scores, tournament_size)
            child_1, child_2 = crossover(parent_1, parent_2)
            child_1 = mutate(child_1)
            child_2 = mutate(child_2)
            new_population.extend([child_1, child_2])
        
        population = new_population

        clear_output(wait = True)
        plt.figure(figsize = (12, 6))
        plt.plot(fitness_history, label = 'Fitness Medio')
        plt.plot(best_fitness_history, label = 'Mejor Fitness')
        plt.grid(linestyle = '--')
        plt.title("Desarrollo por generación")
        plt.xlabel("Generación")
        plt.ylabel("Fitness")
        plt.legend()
        plt.show()
        
    best_individual_features = [
        X_train.columns[i] for i in np.where(best_individual == 1)[0]
        ]
    X_train_best = X_train[best_individual_features]
    X_test_best = X_test[best_individual_features]
    
    if task_type == 'classification' and n_classes == 2:

        model = LGBMClassifier(random_state = random_state, verbose = -1)
        model.fit(X_train_best, y_train)
        y_pred_proba = model.predict_proba(X_test_best)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc(fpr, tpr))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Curva ROC')
        plt.legend(loc="lower right")
        plt.show()

    feature_names = X_train.columns
    selected_features_names = [
        name for 
        name, include in zip(feature_names, best_individual_features) if include
        ]
    
    return selected_features_names