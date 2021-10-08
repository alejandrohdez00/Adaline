import pandas as pd
import numpy as np
from Adaline import Adaline
import matplotlib.pyplot as plt

def error_absoluto(salida_esperada, salida):
    return abs(salida_esperada-salida)

#Cargamos el data set
data_set = pd.read_csv("Concrete_Compressive_Strength_data_set.csv")

#Aleatorización de los datos
random = data_set.sample(frac=1).reset_index(drop=True)

#Dataframe a numpy array
matrix = random.to_numpy()

filas_total, columnas_total = matrix.shape
vector_salidas_esperadas = matrix[:, columnas_total-1]

#Normalización de los datos mediante la formula ValNorma = (ValOriginal - ValMinimo)/(ValMax - ValMaximo)

for i in range(len(vector_salidas_esperadas)):
    vector_salidas_esperadas[i] = (vector_salidas_esperadas[i]-np.min(vector_salidas_esperadas))/(np.max(vector_salidas_esperadas)-np.min(vector_salidas_esperadas))


valores_maximos = np.max(matrix, axis=0)
valores_minimos = np.min(matrix, axis=0)
for i in range(columnas_total-1):
    for j in range(filas_total):
        matrix[j][i] = (matrix[j][i]-valores_minimos[i])/(valores_maximos[i]-valores_minimos[i])

print(matrix)


#Separamos el data set en training, validation y test

train_size = int(filas_total - 0.3*filas_total) #70% de los datos en el set de entrenamiento
validation_size = int(train_size + (filas_total - 0.85*filas_total)) #15% de los datos en el set de validacion
test_size = validation_size + (filas_total - validation_size)

train_set = matrix[:train_size,:columnas_total-1]
validation_set = matrix[train_size:validation_size, :columnas_total-1]
test_set = matrix[validation_size:test_size, :columnas_total-1]

print("Introduzca el valor del learning rate")

learning_rate = float(input())

print("Introduzca el número máximo de ciclos de aprendizaje")

max_cycles = int(input())

adaline = Adaline(columnas_total-1, learning_rate)

cycle_count = 0

error_absoluto_record_t = []  # lista de errores para el entrenamiento
error_absoluto_record_v = []  # Lista de errores para la validacion

while cycle_count<max_cycles:

    error_absoluto_acumulado_t = 0
    error_absoluto_acumulado_v = 0

    #Entrenamiento
    for i in range(len(train_set)):
        adaline.calculoSalida(train_set[i])
        error_absoluto_acumulado_t += error_absoluto(vector_salidas_esperadas[i], adaline.salida)
        adaline.ajustePesos(train_set[i], vector_salidas_esperadas[i])

    error_absoluto_medio_t = error_absoluto_acumulado_t / len(train_set)
    error_absoluto_record_t.append(error_absoluto_medio_t)

    #Validacion
    for i in range(len(validation_set)):
        adaline.calculoSalida(validation_set[i])
        error_absoluto_acumulado_v += error_absoluto(vector_salidas_esperadas[i+train_size], adaline.salida)

    error_absoluto_medio_v = error_absoluto_acumulado_v / len(validation_set)
    error_absoluto_record_v.append(error_absoluto_medio_v)

    cycle_count += 1

#Guardamos modelo
fichero_modelo = open(r"fichero_modelo.txt", "r+")

for i in range(len(adaline.pesos)):
    fichero_modelo.write(f"Peso W{i+1} = {adaline.pesos[i]}\n")
fichero_modelo.write(f"Umbral = {adaline.umbral}")

#Test
error_absoluto_acumulado_test = 0

fichero_salidas = open(r"fichero_salidas.txt", "r+")

for i in range(len(validation_set)):
    adaline.calculoSalida(test_set[i])
    fichero_salidas.write(f"Salida patron {i}: {adaline.salida}\n")
    error_absoluto_acumulado_test += error_absoluto(vector_salidas_esperadas[i+validation_size], adaline.salida)

error_absoluto_medio_test = error_absoluto_acumulado_test / len(test_set)


plt.plot(list(range(max_cycles)), error_absoluto_record_t)
plt.plot(list(range(max_cycles)), error_absoluto_record_v)
print(error_absoluto_medio_test)
plt.show()









