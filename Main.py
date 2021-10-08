import pandas as pd
import numpy as np
from Adaline import Adaline
import matplotlib.pyplot as plt

#Funcion para calcular el error cuadratico
def error_cuadratico(salida_esperada, salida):
    return pow((salida_esperada-salida),2)

#Funcion para normalizar los datos
def minmax_norm(df):
    return (df - df.min()) / (df.max() - df.min())


#Cargamos el data set
data_set = pd.read_csv("Concrete_Compressive_Strength_data_set.csv")

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

filas_total, columnas_total = train.shape


#Dividimos los datos de train, validation y test en informacion de variables, es decir, las variables y salidas esperadas, la última columna

train_set = train[:, :columnas_total-1]
train_salidas = train[:, -1]



validation_set = validation[:, :columnas_total-1]
validation_salidas = validation[:, -1]

test_set = test[:, :columnas_total-1]
test_salidas = test[:, -1]


print(train_set.shape,train_salidas.shape, validation_set.shape, validation_salidas.shape, test_set.shape, test_salidas.shape)
print(len(train_set), len(validation_set), len(test_set))

print("Introduzca el valor del learning rate")

learning_rate = float(input())

print("Introduzca el número máximo de ciclos de aprendizaje")

max_cycles = int(input())

adaline = Adaline(columnas_total-1, learning_rate)

cycle_count = 0

error_cuadratico_record_t = []  # lista de errores para el entrenamiento
error_cuadratico_record_v = []  # Lista de errores para la validacion

while cycle_count<max_cycles:

    error_cuadratico_acumulado_t = 0
    error_cuadratico_acumulado_v = 0

    #Entrenamiento
    for i in range(len(train_set)):
        adaline.calculoSalida(train_set[i])
        error_cuadratico_acumulado_t += error_cuadratico(train_salidas[i], adaline.salida)
        print(f"Salida {i}: {adaline.salida}")
        adaline.ajustePesos(train_set[i], train_salidas[i])

    error_cuadratico_medio_t = error_cuadratico_acumulado_t / len(train_set)
    error_cuadratico_record_t.append(error_cuadratico_medio_t)

    #Validacion
    for i in range(len(validation_set)):
        adaline.calculoSalida(validation_set[i])
        error_cuadratico_acumulado_v += error_cuadratico(validation_salidas[i], adaline.salida)

    error_cuadratico_medio_v = error_cuadratico_acumulado_v / len(validation_set)
    error_cuadratico_record_v.append(error_cuadratico_medio_v)

    print(f"    {cycle_count}   |   {error_cuadratico_medio_t}  |   {error_cuadratico_medio_v}")

    cycle_count += 1

#Guardamos modelo
fichero_modelo = open(r"fichero_modelo.txt", "r+")

for i in range(len(adaline.pesos)):
    fichero_modelo.write(f"Peso W{i+1} = {adaline.pesos[i]}\n")
fichero_modelo.write(f"Umbral = {adaline.umbral}")

#Test
error_cuadratico_acumulado_test = 0

fichero_salidas = open(r"fichero_salidas.txt", "r+")

for i in range(len(test_set)):
    adaline.calculoSalida(test_set[i])
    fichero_salidas.write(f"Salida patron {i}: {adaline.salida}\n")
    error_cuadratico_acumulado_test += error_cuadratico(test_salidas[i], adaline.salida)

error_cuadratico_medio_test = error_cuadratico_acumulado_test / len(test_set)


plt.plot(list(range(max_cycles)), error_cuadratico_record_t, color="red")
plt.plot(list(range(max_cycles)), error_cuadratico_record_v, color="green")
plt.show()
print(error_cuadratico_medio_test)









