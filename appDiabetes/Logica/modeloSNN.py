import os
from django.urls import reverse
import pandas as pd
from sklearn.pipeline import Pipeline
from tensorflow.python.keras.models import load_model, model_from_json
from keras import backend as K
import pickle
import keras

class modeloSNN():
        
    #Función para cargar preprocesador
    def cargarPipeline(self, nombreArchivo):

        with open(nombreArchivo, 'rb') as handle:
            pipeline = pickle.load(handle)
            print("Pickle Cargada desde Archivo") 
        return pipeline
    print("llego aqui pipe")
    
    #Función para cargar red neuronal 
    def cargarNN(self, nombreArchivo):  
        model = keras.models.load_model(nombreArchivo)
        print("Red Neuronal Cargada desde Archivo") 
        return model
    
    def cargar_naive_bayes(self , nombre_archivo):
        with open(nombre_archivo, 'rb') as file:
            modelo_cargado = pickle.load(file)
        return modelo_cargado
    
    print("llego aqui RN")
        #Función para integrar el preprocesador y la red neuronal en un Pipeline
    def cargarModelo(self):
        try:
            print("cargando modelo")
            directorio_actual = os.path.abspath(os.path.dirname(__file__))
            #Se carga el Pipeline de Preprocesamiento
            nombreArchivoPreprocesador= os.path.join(directorio_actual, 'Recursos','pipePreprocesadores.pickle')
            #nombreArchivoPreprocesador='Recursos/pipePreprocesadores.pickle'
            print(nombreArchivoPreprocesador)
            pipe=self.cargarPipeline(self, nombreArchivoPreprocesador)
            print('Pipeline de Preprocesamiento Cargado')
            cantidadPasos=len(pipe.steps)
            print("Cantidad de pasos: ", cantidadPasos)
            print(pipe.steps)
            #Se carga la Red Neuronal
            modeloOptimizado=self.cargarNN(self , os.path.join(directorio_actual, 'Recursos', 'modeloRedNeuronalBase.h5'))
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
    

        #Esta es la función para calcular la certeza (confianza o probabilidad) asociada a la predicción de clase
    def obtenerResultadosyCertezas(lista):
        predicciones=lista
        marcas=[]
        certezas=[]
        nuevomax=1
        nuevomin=0
        marca=-1
        certeza=-1
        for i in range(len(lista)):
            prediccion=lista[i]
            if (prediccion < 0.5):
                marca = 'Con base en los datos proporcionados, puedo confirmar que la progresión de la enfermedad en su caso ''NO'' indica signos adversos. Puede estar tranquilo/a, ya que no se observan elementos que sugieran una evolución desfavorable de la diabetes en su salud. La certeza de esta afirmación alcanza un porcentaje elevado, situándose en:'
                maxa=0.5
                mina=0
                certeza=1-((prediccion-mina)/(maxa-mina)*(nuevomax-nuevomin)+nuevomin)
                certeza=str(int((certeza)*100))+'%'
            elif (prediccion >= 0.5):
                marca = 'Con base en los datos proporcionados, puedo confirmar que la progresión de la enfermedad en su caso ''SI'' indica signos adversos. Se recomienda que consulte con un profesional de la salud para obtener una evaluación precisa de la situación y tomar las medidas necesarias. La certeza de esta observación alcanza un porcentaje significativo de:'
                maxa=1
                mina=0.5
                certeza=(prediccion-mina)/(maxa-mina)*(nuevomax-nuevomin)+nuevomin
                certeza=str(int((certeza)*100))+'%'
            marcas.append(marca)
            certezas.append(certeza)
        return prediccion, marcas, certezas
    




    def predecirNuevoCliente(self, edad, sexo, indice_masa_corporal, precion_arterial, suero_1, suero_2,
                                suero_3, suero_4, suero_5, suero_6, progresion_enfermedad):
            print("EMPIEZA A PREDECIR EL MODELO")
            pipe = modeloSNN.cargarModelo(self)
            cnames = [
                'EDAD', 'SEXO', 'INDICE_MASA_CORPORAL', 'PRECION_ARTERIAL', 'SUERO_1', 'SUERO_2',
                'SUERO_3', 'SUERO_4', 'SUERO_5', 'SUERO_6', 'PROGRESION_ENFERMEDAD'
            ]

            Xnew = [edad, sexo, indice_masa_corporal, precion_arterial, suero_1, suero_2,
                    suero_3, suero_4, suero_5, suero_6, progresion_enfermedad]

            # Xnew_Dataframe = pd.DataFrame(data=[Xnew], columns=cnames)
            # print(Xnew_Dataframe)
            # pred = pipe.predict(Xnew_Dataframe)
            # pred_proba = pred.flatten()[0]  # de 2D a 1D


            Xnew_Dataframe = pd.DataFrame(data=[Xnew],columns=cnames)
            y_pred=pipe.predict(Xnew_Dataframe)[0].tolist()
            predicciones, marcas, certezas= modeloSNN.obtenerResultadosyCertezas(y_pred)
            dataframeFinal_pred=pd.DataFrame({'Resultado':marcas , 'Certeza': certezas})


            resultado = dataframeFinal_pred.to_string(index=False , header=False)

            # print("Probabilidades de predicción:", pred_proba)


            return resultado
    

    print("llego aqui NB Modelo PIPE ")
        #Función para integrar el preprocesador y la red neuronal en un Pipeline
    def cargarPipeNB(self):
        try:
            print(" PIPE cargando modelo AHORA  NB NB NB NB ")
            directorio_actual = os.path.abspath(os.path.dirname(__file__))
            #Se carga el Pipeline de Preprocesamiento
            nombreArchivoPreprocesador= os.path.join(directorio_actual, 'Recursos','pipePreprocesadores.pickle')
            #nombreArchivoPreprocesador='Recursos/pipePreprocesadores.pickle'
            print(nombreArchivoPreprocesador)
            pipe=self.cargarPipeline(self, nombreArchivoPreprocesador)
            print('Pipeline de Preprocesamiento Cargado')
            cantidadPasos=len(pipe.steps)
            print("Cantidad de pasos: ", cantidadPasos)
            print(pipe.steps)
            cantidadPasos=len(pipe.steps)
            print("Cantidad de pasos: ",cantidadPasos)
            print(pipe.steps)
            print('PIPE PARA NB CARGADO CORRECTAMENTE')
            return pipe
        except FileNotFoundError as e:
            print(f"Error archivo no encontrado: {e.filename}")
        except Exception as e:
            print(f"Error inesperado: {e}")
        return None
    




    print("llego aqui NB Modelo ")
        #Función para integrar el preprocesador y la red neuronal en un Pipeline
    def cargarModeloNB(self):
        try:
            print(" PIPE cargando modelo AHORA  NB NB NB NB ")
            directorio_actual = os.path.abspath(os.path.dirname(__file__))
            #Se carga el modelo
            modeloOptimizado=self.cargar_naive_bayes(self , os.path.join(directorio_actual, 'Recursos', 'modeloNaiveBayesBase.pkl'))
            #Se integra la Red Neuronal al final del Pipeline
            return modeloOptimizado
        except FileNotFoundError as e:
            print(f"Error archivo no encontrado: {e.filename}")
        except Exception as e:
            print(f"Error inesperado: {e}")
        return None
    




    def predecirNuevoPacienteNB(
        self, edad, sexo, indice_masa_corporal, precion_arterial,
        suero_1, suero_2, suero_3, suero_4, suero_5, suero_6, progresion_enfermedad
    ):
        print("NB NB entro el Paciente")

        cnames = [
        'EDAD', 'SEXO', 'INDICE_MASA_CORPORAL', 'PRECION_ARTERIAL', 'SUERO_1', 'SUERO_2',
        'SUERO_3', 'SUERO_4', 'SUERO_5', 'SUERO_6', 'PROGRESION_ENFERMEDAD'
    ]


        Xnew=[ edad, sexo, indice_masa_corporal, precion_arterial, suero_1, suero_2,
            suero_3, suero_4, suero_5, suero_6, progresion_enfermedad]

        pipeNB = modeloSNN.cargarPipeNB(self)

        Xnew_Dataframe = pd.DataFrame(data=[Xnew],columns=cnames)
        #pipe=cargarPipeline("pipePreprocesadores")
        Xnew_Transformado=pipeNB.transform(Xnew_Dataframe)
        modelo=modeloSNN.cargarModeloNB(self)

        y_pred=modelo.predict(Xnew_Transformado)
        predicciones, marcas, certezas= modeloSNN.obtenerResultadosyCertezas(y_pred)
        dataframeFinal_pred=pd.DataFrame({'Resultado':marcas , 'Certeza': certezas})


        resultado = dataframeFinal_pred.to_string(index=False , header=False)

        return resultado
