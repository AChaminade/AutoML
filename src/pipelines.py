# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 20:01:28 2024

@author: adrgc
"""

# Librerías
#-----------------------------------------------------------------------------#
# Librerías de tratamiento de datos
from joblib import dump

# Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

#=============================================================================#

class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y = None):
        return self

    def transform(self, X):
        return X[self.columns]

def crear_y_guardar_pipeline(columnas, modelo, tipo_modelo, ruta_guardado, estandarizador = None):
    """
    Crea una pipeline con un selector de columnas, un estandarizador opcional,
    un modelo dado (preentrenado), guarda el tipo de modelo como un parámetro, 
    y guarda la pipeline.
    
    :param columnas: Lista de columnas a seleccionar.
    :param modelo: Modelo de Scikit-Learn ya entrenado.
    :param tipo_modelo: String 'clasificacion' o 'regresion' que indica el tipo del modelo.
    :param ruta_guardado: Ruta para guardar la pipeline.
    :param estandarizador: Estandarizador preentrenado (opcional).
    :return: La pipeline.
    """
    pasos = [('selector', ColumnSelector(columns=columnas))]

    if estandarizador is not None:
        pasos.append(('estandarizador', estandarizador))

    pasos.append(('modelo', modelo))
    pipeline = Pipeline(pasos)

    # Guardar el tipo de modelo como un atributo de la pipeline
    pipeline.tipo_modelo = tipo_modelo

    # Guardar la pipeline
    dump(pipeline, ruta_guardado)

    return pipeline

def cargar_y_ejecutar_pipeline(ruta_pipeline, X_nuevos_datos, obtener_probabilidades=False):
    """
    Carga una pipeline guardada, la ejecuta en un nuevo conjunto de datos y, 
    si es un modelo de clasificación,
    ofrece la opción de devolver probabilidades o predicciones.
    
    :param ruta_pipeline: Ruta de la pipeline guardada.
    :param X_nuevos_datos: Nuevo conjunto de datos para hacer predicciones.
    :param obtener_probabilidades: Si se deben devolver las probabilidades (solo para clasificación).
    :return: Predicciones o probabilidades.
    """
    # Cargar la pipeline
    pipeline_cargada = load(ruta_pipeline)

    # Determinar el tipo de modelo desde el atributo guardado
    tipo_modelo = pipeline_cargada.tipo_modelo

    # Ejecutar la pipeline
    if tipo_modelo == 'clasificacion' and obtener_probabilidades:
        return pipeline_cargada.predict_proba(X_nuevos_datos)
    else:
        return pipeline_cargada.predict(X_nuevos_datos)