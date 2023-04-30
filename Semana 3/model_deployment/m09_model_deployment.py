#!/usr/bin/python


import pandas as pd
import joblib
import sys
import os

def predict_price(Diccionario):
    Particiones = [0.2,0.4,0.6,0.8,0.9,0.95,0.985,1]
    Particiones_marcas_df = pd.read_excel('Particiones_marcas.xlsx')
    #Transformacion para manejar mismo lenguaje
    Particiones_marcas_df= Particiones_marcas_df[['Make','Model','Particion']]
    Particiones_Marcas = {}
    for i in Particiones_marcas_df['Particion'].drop_duplicates().to_numpy():
        Particiones_Marcas[i]= Particiones_marcas_df\
            [Particiones_marcas_df['Particion']==i][['Make','Model']]
    #
    #Revisado
    
    df_api = pd.DataFrame(Diccionario)
    #Transformación
    df_api = df_api.reset_index().rename(columns={'index':'ID'})
    df_models_test = {}
    Particiones_api=[]
    suma=0
    for i in Particiones:
        df_aux=pd.merge(Particiones_Marcas[i],df_api,on=['Make','Model'])
        if df_aux.shape[0] > 0 :
            #print(i)
            Particiones_api.append(i)
            df_aux['Make-Mod'] = \
                df_aux.apply(lambda x: x['Make']+x['Model'], axis=1)
            df_aux=df_aux.drop(columns=['Make','Model'],axis=1)
            df_models_test[i]=df_aux
            suma += df_aux.shape[0]
    #Traerse las columnas de las particiones seleccionadas:
    columnas = {}
    for i in Particiones_api:
        columnas[i] = pd.read_csv('Columnas Particion ' + str(i) + '.csv')['Column']
    
    #Escalado
    #Antes debemos cargar los escaladores correspondientes:
    scalers = {}
    for i in Particiones_api:
        scalers[i] = joblib.load(os.path.dirname(__file__) + '/scalers '+ str(i)+'.pkl') 
    
    #
    Variable_ID = ['ID']
    df_scaled_test = {}
    Variable_y = ['Price']
    Variables_numericas=['Year','Mileage']
    Variables_categoricas= ['State','Make-Mod']
    for i in Particiones_api:
        df_model_nums = df_models_test[i][Variables_numericas]
        scaled_data = scalers[i].transform(df_model_nums.to_numpy())
        scaled_data = pd.DataFrame(scaled_data,columns=['Year','Mileage'])
        df_scaled_test[i]=pd.concat([df_models_test[i][Variable_ID+Variables_categoricas],scaled_data],axis=1)

    #Codificación ONE-HOT
    df_dummies_test = {}
    for i in Particiones_api:
        df_dummies_test[i]=pd.get_dummies(df_scaled_test[i],columns=Variables_categoricas)

    A_Predecir = {}
    for i in Particiones_api:
        Clave = {}
        Clave['xVal']= df_dummies_test[i].drop('ID',axis=1)
        Clave['id'] = df_dummies_test[i]['ID']
        A_Predecir[i]=Clave

    #Revisión de coherencia de columnas
    for i in Particiones_api:
        Columnas_Val = A_Predecir[i]['xVal'].columns.tolist()
        Columnas_Train = columnas[i].tolist()
        for k in Columnas_Train:
            if k not in Columnas_Val:
                A_Predecir[i]['xVal'][k]=0
        A_Predecir[i]['xVal'] = A_Predecir[i]['xVal'].reindex(columns=Columnas_Train)
        Columnas_Val = A_Predecir[i]['xVal'].columns.tolist()
        print('Las columnas en la partición ' +str(i) +' son las mismas de entrenamiento: '+str(Columnas_Val==Columnas_Train))
    
    predictor_a_utilizar = {}
    for i in Particiones_api:
        predictor_a_utilizar[i] = \
            joblib.load(os.path.dirname(__file__) + '/Precios_Vehiculos '+ str(i)+'.pkl') 
    
    Predicciones={}
    for i in Particiones_api:
        yPred= predictor_a_utilizar[i].predict(A_Predecir[i]['xVal'])
        yIndex = A_Predecir[i]['id']
        Predicciones[i]=pd.DataFrame(zip(yIndex,yPred),columns=['ID','Price'])
    
    Prediccion_api = pd.concat(Predicciones.values()).sort_values('ID').set_index('ID')

    return Prediccion_api


if __name__ == "__main__":
    
    if len(sys.argv) == 1:
        print('Please add an Diccionario')

        
    else:

        Diccionario = sys.argv[1]

        p1 = predict_price(Diccionario)
        
        print(Diccionario)
        print('Probability of Phishing: ')
        