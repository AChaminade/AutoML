# -*- coding: utf-8 -*-
"""
Created on Thu Jan 18 18:51:05 2024

@author: adrgc
"""

# Librerías
#-----------------------------------------------------------------------------#
# Librerías ML
from sklearn.model_selection import cross_val_score

#=============================================================================#

def select_best_model(
        models, 
        X_train, 
        y_train, 
        cat_cols = [], 
        random_state = 42
        ):
    if cat_cols:
        models = models[-2:]

    best_score = 0
    best_model = None

    for model in models:
        # Validación cruzada
        if 'CatBoost' in model.__name__:
            model_ = model(
                cat_features = cat_cols, 
                verbose = 0, 
                random_state = random_state
                )
        else:
            try:
                model_ = model(verbose = -1, random_state = random_state)
            except:
                model_ = model()
        
        scores = cross_val_score(model_, X_train, y_train, cv = 5)
        mean_score = scores.mean()
        print(f"Modelo: {model.__name__}, Puntuación Media: {mean_score}")

        if mean_score > best_score:
            best_score = mean_score
            best_model = model

    return best_model.__name__