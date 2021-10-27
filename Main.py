import pandas as pd
import numpy as np
import copy
from Adaline import Adaline
import matplotlib.pyplot as plt

#Funcion para calcular el error cuadratico
def error_cuadratico(salida_esperada, salida):
    return pow((salida_esperada-salida),2)

#Funcion para normalizar los datos
def minmax_norm(df):
    return (df - df.min()) / (df.max() - df.min())

#Funcion para desnormalizar
def desnorm(norm_val, min, max):
    return norm_val * (max - min) + min


#Cargamos el data set
data_set = pd.read_csv("Concrete_Compressive_Strength_data_set.csv")
print(data_set.describe())

#Calculamos el mínimo y el máximo de la columna de las salidas antes de normalizar, para poder desnormalizar la salida en el futuro
min_salida = data_set["ConcreteCompressiveStrength"].min()
max_salida = data_set["ConcreteCompressiveStrength"].max()


#Normalización de los datos
data_set_norm = minmax_norm(data_set)

#Aleatorización de los datos
#np.random.seed(0)
randomized_data_set = data_set_norm.sample(frac=1).reset_index(drop=True)


#Separamos los datos en train, validation y test
train, validation, test = np.split(randomized_data_set, [int(0.7*len(randomized_data_set)), int(0.85*len(randomized_data_set))])

#Exportamos los datos para usarlos en el Perceptron Multicapa
train.to_csv(r'train.csv', index = False)
validation.to_csv(r'valid.csv', index = False)
test.to_csv(r'test.csv', index = False)


#Dataframe a numpy array para mejor tratamiento

train = train.to_numpy()
validation = validation.to_numpy()
test = test.to_numpy()


#Dividimos los datos de train, validation y test en informacion de variables, es decir, las variables y salidas esperadas, la última columna

columnas_total = train.shape[1]

train_set = train[:, :columnas_total-1]
train_salidas_deseadas = train[:, columnas_total-1]

validation_set = validation[:, :columnas_total-1]
validation_salidas_deseadas = validation[:, columnas_total-1]

test_set = test[:, :columnas_total-1]
test_salidas_deseadas = test[:, columnas_total-1]

print("Introduzca el valor del learning rate")

learning_rate = float(input())

print("Introduzca el número máximo de ciclos de aprendizaje")

max_cycles = int(input())

adaline = Adaline(columnas_total-1, learning_rate)

cycle_count = 0

error_cuadratico_record_t = []  # lista de errores para el entrenamiento
error_cuadratico_record_v = []  # Lista de errores para la validacion

#Para poder comprobar y guardar el mejor modelo creamos un objeto adaline y una variable de comprobacion con un valor grande

best_adaline = Adaline(columnas_total-1, learning_rate)
min_error_val = 5

while cycle_count < max_cycles:

    error_cuadratico_acumulado_t = 0
    error_cuadratico_acumulado_v = 0

    #Entrenamiento
    for i in range(len(train_set)):
        adaline.calculoSalida(train_set[i])
        adaline.ajustePesos(train_set[i], train_salidas_deseadas[i])

    #Calculamos error entrenamiento
    for i in range(len(train_set)):
        adaline.calculoSalida(train_set[i])
        error_cuadratico_acumulado_t += error_cuadratico(train_salidas_deseadas[i], adaline.salida)

    error_cuadratico_medio_t = error_cuadratico_acumulado_t / len(train_set)
    error_cuadratico_record_t.append(error_cuadratico_medio_t)

    #Calculamos error validacion
    for i in range(len(validation_set)):
        adaline.calculoSalida(validation_set[i])
        error_cuadratico_acumulado_v += error_cuadratico(validation_salidas_deseadas[i], adaline.salida)

    error_cuadratico_medio_v = error_cuadratico_acumulado_v / len(validation_set)
    error_cuadratico_record_v.append(error_cuadratico_medio_v)

    #Comprobamos si es mejor modelo
    if(error_cuadratico_medio_v < min_error_val):
        min_error_val = error_cuadratico_medio_v #Guardamos el minimo error de validacion hasta el momento
        best_adaline = copy.deepcopy(adaline) #Guardamos modelo


    #Printeamos información útil
    print(f"Ciclo: {cycle_count + 1}")
    print(adaline)
    print(f"Error Entrenamiento: {error_cuadratico_medio_t}")
    print(f"Error Validación: {error_cuadratico_medio_v} \n")

    cycle_count += 1


#Guardamos modelo
fichero_modelo = open(r"fichero_modelo.txt", "w")
fichero_modelo.write(str(best_adaline))

#FIN ADALINE

#**********************************************************************************
#Creacion y desarollo de los ficheros
#*********************************************************************************************

#Calculamos el error de test con el mejor modelo obtenido y hacemos fichero de salidas obtenidas
error_cuadratico_acumulado_test = 0

fichero_salidas = open(r"fichero_salidas.txt", "w")

salidas_desnorm = [] #Almacenamos las salidad desnormalizadas obtenidas en el test

for i in range(len(test_set)):
    best_adaline.calculoSalida(test_set[i])
    s_desnorm = desnorm(best_adaline.salida, min_salida, max_salida)
    fichero_salidas.write(f"Salida patron {i+1}: {desnorm(s_desnorm, min_salida, max_salida)}\n")
    salidas_desnorm.append(s_desnorm)
    error_cuadratico_acumulado_test += error_cuadratico(test_salidas_deseadas[i], best_adaline.salida)

error_cuadratico_medio_test = error_cuadratico_acumulado_test / len(test_set)



#Fichero salidas deseadas test
fichero_salidas_deseadas = open(r"fichero_salidas_deseadas.txt", "w")
salidas_deseadas_desnorm = []

for i in range(len(test_salidas_deseadas)):
    s_deseada_desnorm = desnorm(test_salidas_deseadas[i], min_salida, max_salida)
    fichero_salidas_deseadas.write(f"Salida deseada patron {i+1}: {s_deseada_desnorm}\n")
    salidas_deseadas_desnorm.append(s_deseada_desnorm)


#Fichero que contenga la evolución del error en entrenamiento y validación del mejor experimento.
fichero_evolucion_error = open(r"fichero_evolucion_error.txt", "w")


cycle_count = 0

while cycle_count < max_cycles:

    error_cuadratico_acumulado_t = 0
    error_cuadratico_acumulado_v = 0

    for i in range(len(train_set)):
        adaline.calculoSalida(train_set[i])
        error_cuadratico_acumulado_t += error_cuadratico(train_salidas_deseadas[i], best_adaline.salida)
    error_cuadratico_medio_t = error_cuadratico_acumulado_t / len(train_set)

    for i in range(len(validation_set)):
        best_adaline.calculoSalida(validation_set[i])
        error_cuadratico_acumulado_v += error_cuadratico(validation_salidas_deseadas[i], best_adaline.salida)
    error_cuadratico_medio_v = error_cuadratico_acumulado_v / len(validation_set)

    fichero_evolucion_error.write(f"Ciclo{cycle_count + 1}:\nError entrenamiento: {error_cuadratico_medio_t}\nError validacion: {error_cuadratico_medio_v}\n")

    cycle_count += 1
#***************************************************************************************************
#Gráficas
#***************************************************************************************************

#Grafica Error entrenamiento y validacion
fig, ax = plt.subplots()
ax.plot(list(range(max_cycles)), error_cuadratico_record_t, label='Error ent.')
ax.plot(list(range(max_cycles)), error_cuadratico_record_v, label='Error val.')
ax.set_xlabel('Ciclos')
ax.set_ylabel('Error')
ax.set_title("Entrenamiento vs Validacion")
ax.legend()
fig.show()

#Grafica salida obtenida y salida deseada
fig1, ax1 = plt.subplots()
ax1.plot(list(range(len(salidas_deseadas_desnorm))), salidas_deseadas_desnorm, label='Salidas deseadas')
ax1.plot(list(range(len(salidas_desnorm))), salidas_desnorm, label='Salidas obtenidas')
ax1.set_xlabel('Patrones')
ax1.set_ylabel('MPa')
ax1.set_title("Salidas deseadas vs obtenidas")
ax1.legend()
fig1.show()

print("*******************************************")
print(f"Error test: {error_cuadratico_medio_test}")










