#!/usr/bin/python


import pandas as pd
import joblib
import sys
import os

import keras

from keras.models import Sequential
from keras.layers import Dense,Dropout

from keras.activations import sigmoid
from sentence_transformers import SentenceTransformer

def clasif_genre(plot):
    
    #Generación de red neuronal
    keras.backend.clear_session()
    model = Sequential()
    model.add(Dense(689, input_shape=(384,),activation='swish')) 
    model.add(Dropout(0.05))
    model.add(Dense(128, activation='sigmoid'))
    model.add(Dropout(0.4))
    model.add(Dense(24, activation='sigmoid'))
    #Carga de Pesos
    model.load_weights('Api_pesos_modelo_calibrado.h5')


    model_all_Mini = joblib.load('Api_AllMiniModel.pkl')
    sentences = [plot]
    embeddings_plot= model_all_Mini.encode(sentences)
    y_pred_plot= model.predict(embeddings_plot)

    cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
    'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
    'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']

    df_aux = pd.DataFrame(y_pred_plot,columns=cols).T.sort_values(by=0,ascending=False).rename(columns={0:'prob'})
    genres_mas_probables = df_aux[df_aux['prob']>0.5].to_dict()

    return genres_mas_probables


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add a plot')
        
    else:
        Diccionario = sys.argv[1]

        p1 = clasif_genre(Diccionario)
        
        print(Diccionario)
        print('Probability of Phishing: ')
        