# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:15:45 2024

@author: adrgc
"""

# Librerías
#-----------------------------------------------------------------------------#
# Modelos de clasificación
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import BaggingClassifier

# Regresión
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import BaggingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

#=============================================================================#

classification = [
    LogisticRegression,
    SVC,
    DecisionTreeClassifier,
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    KNeighborsClassifier,
    GaussianNB,
    MultinomialNB,
    BernoulliNB,
    Perceptron,
    MLPClassifier,
    SGDClassifier,
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
    BaggingClassifier,
    XGBClassifier,
    LGBMClassifier,
    CatBoostClassifier
    ]

regression = [
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    SVR,
    DecisionTreeRegressor,
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    KNeighborsRegressor,
    SGDRegressor,
    BaggingRegressor,
    XGBRegressor,
    LGBMRegressor,
    CatBoostRegressor
    ]