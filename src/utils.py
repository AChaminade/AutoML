# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 13:56:17 2024

@author: adrgc
"""

def identify_columns(df):
    numeric_columns = []
    categoric_columns = []

    for columna in df.columns:
        es_numerica = True
        for valor in df[columna]:
            try:
                # Intenta convertir a float
                float(valor)
            except ValueError:
                # Si falla, la columna no es numérica
                es_numerica = False
                break

        if es_numerica:
            numeric_columns.append(columna)
        else:
            categoric_columns.append(columna)

    return numeric_columns, categoric_columns