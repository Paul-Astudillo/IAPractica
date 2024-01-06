import sys
sys.path.append(".")

from django.shortcuts import render
from appDiabetes.Logica.modeloSNN import modeloSNN #para utilizar el método inteligente
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from appDiabetes.Logica import modeloSNN
import json
from django.http import JsonResponse


class Clasificacion():
    def determinarAprobacion(request):
        return render(request, "formulario.html")
    
    @api_view(['GET','POST'])
    def predecir(request):
        try:
            #Formato de datos de entrada
            print("Entro al request")
            edad = int(''+request.POST.get('edad'))
            print(edad)

            sexo=request.POST.get('sexo')
            print(sexo)

            indice_masa_corporal = float(request.POST.get('indice_masa_corporal', 0.0))
            print(indice_masa_corporal)

            precion_arterial = float(request.POST.get('precion_arterial', 0.0))
            print(precion_arterial)

            suero_1 = float(request.POST.get('suero_1', 0.0))
            print(suero_1)

            suero_2 = float(request.POST.get('suero_2', 0.0))
            print(suero_2)

            suero_3 = float(request.POST.get('suero_3', 0.0))
            print(suero_3)

            suero_4 = float(request.POST.get('suero_4', 0.0))
            print(suero_4)

            suero_5 = float(request.POST.get('suero_5', 0.0))
            print(suero_5)

            suero_6 = float(request.POST.get('suero_6', 0.0))
            print(suero_6)

            progresion_enfermeda = float(request.POST.get('progresion_enfermeda', 0.0))
            print(progresion_enfermeda)


            modelo=request.POST.get('modelo')
            print("Modelo seleccionado",modelo)

            #Consumo de la lógica para predecir si se aprueba o no el crédito
            print("leido todos los datos")
            print("Creo la clase modeloSNN")

            #RedNeuronal
            resul=modeloSNN.modeloSNN.predecirNuevoCliente(modeloSNN.modeloSNN, edad=edad, sexo=sexo, indice_masa_corporal=indice_masa_corporal, precion_arterial=precion_arterial,
            suero_1=suero_1, suero_2=suero_2, suero_3=suero_3, 
            suero_4=suero_4, suero_5=suero_5, suero_6=suero_6, 
            progresion_enfermedad=progresion_enfermeda)

            #NBayes
        except Exception as e:
            print(e)
            resul='Datos inválidos'
           
        return render(request, "resultado.html",{"e":resul})
    

        
    @api_view(['GET','POST'])
    def predecirNB(request):
        try:
            #Formato de datos de entrada
            print("Entro al request")
            edad = int(''+request.POST.get('edad'))
            print(edad)

            sexo=request.POST.get('sexo')
            print(sexo)

            indice_masa_corporal = float(request.POST.get('indice_masa_corporal', 0.0))
            print(indice_masa_corporal)

            precion_arterial = float(request.POST.get('precion_arterial', 0.0))
            print(precion_arterial)

            suero_1 = float(request.POST.get('suero_1', 0.0))
            print(suero_1)

            suero_2 = float(request.POST.get('suero_2', 0.0))
            print(suero_2)

            suero_3 = float(request.POST.get('suero_3', 0.0))
            print(suero_3)

            suero_4 = float(request.POST.get('suero_4', 0.0))
            print(suero_4)

            suero_5 = float(request.POST.get('suero_5', 0.0))
            print(suero_5)

            suero_6 = float(request.POST.get('suero_6', 0.0))
            print(suero_6)

            progresion_enfermeda = float(request.POST.get('progresion_enfermeda', 0.0))
            print(progresion_enfermeda)

            modelo=request.POST.get('modelo')
            print("Modelo seleccionado",modelo)

            #Consumo de la lógica para predecir si se aprueba o no el crédito
            print("leido todos los datos")
            print("Creo la clase modeloSNN")

            print("NB   NB   NB   NB  NB  NB ")
            #RedNeuronal
            resul=modeloSNN.modeloSNN.predecirNuevoPacienteNB(modeloSNN.modeloSNN, edad=edad, sexo=sexo, indice_masa_corporal=indice_masa_corporal, precion_arterial=precion_arterial,
            suero_1=suero_1, suero_2=suero_2, suero_3=suero_3, 
            suero_4=suero_4, suero_5=suero_5, suero_6=suero_6, 
            progresion_enfermedad=progresion_enfermeda)

            #NBayes
        except Exception as e:
            print(e)
            resul='Datos inválidos'
           
        return render(request, "resultadoNB.html",{"e":resul})





@csrf_exempt
@api_view(['GET','POST'])
def predecirIOJson(request):
        print(request)
        print('***********************************************')
        print(request.body)
        print('***********************************************')
        body = json.loads(request.body.decode('utf-8'))
        #Formato de datos de entrada
        PLAZOS = int(body.get("PLAZOMESESCREDITO"))
        MONTOCREDITO = float(body.get("MONTOCREDITO"))
        TASAPAGO = float(body.get("TASAPAGO"))
        EDAD = int(body.get("EDAD"))
        CANTIDADPERSONASAMANTENER= int(body.get("CANTIDADPERSONASAMANTENER"))
        EMPLEO=str(body.get("EMPLEO"))
        print(PLAZOS)
        print(MONTOCREDITO)
        print(TASAPAGO)
        print(EDAD)
        print(CANTIDADPERSONASAMANTENER)
        print(EMPLEO)
        modelo = modeloSNN()
        resul = modelo.predecirNuevoCliente(modeloSNN,PLAZOMESESCREDITO=PLAZOS,MONTOCREDITO=MONTOCREDITO,TASAPAGO=TASAPAGO,EDAD=EDAD,CANTIDADPERSONASAMANTENER=CANTIDADPERSONASAMANTENER,EMPLEO=EMPLEO)  
        
        data = {'result': resul}
        resp=JsonResponse(data)
        resp['Access-Control-Allow-Origin'] = '*'
        return resp
