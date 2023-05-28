#!/usr/bin/python

import pandas as pd
import numpy as np
import joblib
import sys
import os
from xgboost import XGBClassifier
from sklearn.multiclass import OneVsRestClassifier

def clasif_genre(text):
    
    dataTesting= pd.DataFrame([text],columns=['plot'])

    #Funciones necesarias
    def minusculizar(df):
        df2 = df.copy()
        df2['plot'] = df2['plot'].apply(lambda x : x.lower())
        return df2

    def lematizar_texto(texto, diccionario):
        palabras = texto.split()  # Dividir el texto en palabras
        palabras_lematizadas = [diccionario.get(palabra, palabra) for palabra in palabras]  # Obtener las palabras lematizadas del diccionario
        palabras_filtradas = [palabra for palabra in palabras_lematizadas if palabra.lower() not in stopwords_english]  # Filtrar las stopwords
        texto_lematizado = ' '.join(palabras_filtradas)  # Unir las palabras lematizadas en un nuevo texto
        return texto_lematizado

    #Carga de archivos necesarios
    diccionario_original_a_lemas = joblib.load('Diccionario_con_lemas.pkl')
    vectorizer = joblib.load('API_vectorizer.pkl')
    model = joblib.load('XGBoost clf.pkl')
    stopwords_english = joblib.load('stopwords_english.pkl')
    #Transformaciones
    dataTesting_min  = minusculizar(dataTesting)
    dataTesting_min['plot_lematized'] = dataTesting_min['plot'].apply(lambda x: lematizar_texto(x,diccionario_original_a_lemas))

    plot_vectorizer_test = vectorizer.transform(dataTesting_min['plot_lematized'])
    y_pred_plot=model.predict_proba(plot_vectorizer_test)

    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    df_aux = pd.DataFrame(y_pred_plot,columns=cols).T.sort_values(by=0,ascending=False).rename(columns={0:'prob'})
    genres_mas_probables = df_aux[df_aux['prob']>0.5].to_dict()['prob']
    return genres_mas_probables


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add a plot')
        
    else:
        Diccionario = sys.argv[1]

        p1 = clasif_genre(Diccionario)
        
        print(Diccionario)
        print('Probability of Genre: ')
        