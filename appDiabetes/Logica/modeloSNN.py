import os
from django.urls import reverse
import pandas as pd
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model, model_from_json
from keras import backend as K
import pickle
import keras

class modeloSNN:
        

    def __init__(self) -> None:
        print("Clase modelo Preprocesamiento y SNN")

    #Función para cargar preprocesador
    def cargarPipeline(self, nombreArchivo):

        with open(nombreArchivo, 'rb') as handle:
            pipeline = pickle.load(handle)
            print("Pickle Cargada desde Archivo") 
        return pipeline
    
    #Función para cargar red neuronal 
    def cargarNN(self, nombreArchivo):  
        model = keras.models.load_model(nombreArchivo)
        print("Red Neuronal Cargada desde Archivo") 
        return model
        #Función para integrar el preprocesador y la red neuronal en un Pipeline
    def cargarModelo(self):
        try:
            print("cargando modelo")
            directorio_actual = os.path.abspath(os.path.dirname(__file__))
            #Se carga el Pipeline de Preprocesamiento
            nombreArchivoPreprocesador= os.path.join(directorio_actual, 'Recursos','pipePreprocesadores.pickle')
            #nombreArchivoPreprocesador='Recursos/pipePreprocesadores.pickle'
            print(nombreArchivoPreprocesador)
            pipe=self.cargarPipeline(nombreArchivoPreprocesador)
            print('Pipeline de Preprocesamiento Cargado')
            cantidadPasos=len(pipe.steps)
            print("Cantidad de pasos: ", cantidadPasos)
            print(pipe.steps)
            #Se carga la Red Neuronal
            modeloOptimizado=self.cargarNN(os.path.join(directorio_actual, 'Recursos', 'modeloRedNeuronalBase.h5'))
            #Se integra la Red Neuronal al final del Pipeline
            pipe.steps.append(['modelNN',modeloOptimizado])
            cantidadPasos=len(pipe.steps)
            print("Cantidad de pasos: ",cantidadPasos)
            print(pipe.steps)
            print('Red Neuronal integrada al Pipeline')
            return pipe
        except FileNotFoundError as e:
            print(f"Error archivo no encontrado: {e.filename}")
        except Exception as e:
            print(f"Error inesperado: {e}")
        return None
@staticmethod
def predecirNuevoCliente(edad=24.0, sexo="1", indice_masa_corporal=25.3, precion_arterial=84.0, suero_1=198.0, suero_2=131.4,
                             suero_3=40.0, suero_4=5.00, suero_5=4.8903, suero_6=89.0, progresion_enfermedad=206.0):
        print("EMPIEZA A PREDECIR EL MODELO")
        pipe = modeloSNN.cargarModelo()
        cnames = [
            'EDAD', 'SEXO', 'INDICE_MASA_CORPORAL', 'PRECION_ARTERIAL', 'SUERO_1', 'SUERO_2',
            'SUERO_3', 'SUERO_4', 'SUERO_5', 'SUERO_6', 'PROGRESION_ENFERMEDAD'
        ]

        Xnew = [edad, sexo, indice_masa_corporal, precion_arterial, suero_1, suero_2,
                suero_3, suero_4, suero_5, suero_6, progresion_enfermedad]

        Xnew_Dataframe = pd.DataFrame(data=[Xnew], columns=cnames)
        print(Xnew_Dataframe)
        pred = (pipe.predict(Xnew_Dataframe) > 0.5).astype("int32")
        print(pred)
        pred = pred.flatten()[0]  # de 2D a 1D
        if pred == 1:
            pred = ' SI , Tiene Diabetes'
        else:
            pred = 'NO , Tiene Diabetes'
        return pred