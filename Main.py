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



matrix = random.to_numpy()

vector_salidas_esperadas = matrix[:, -1]
matrix = np.delete(matrix, -1, 1)

num_entradas, num_atributos = matrix.shape

#Normalización de los datos mediante la formula ValNorma = (ValOriginal - ValMinimo)/(ValMax - ValMaximo)

for i in range(len(vector_salidas_esperadas)):
    vector_salidas_esperadas[i] = (vector_salidas_esperadas[i]-np.min(vector_salidas_esperadas))/(np.max(vector_salidas_esperadas)-np.min(vector_salidas_esperadas))


valores_maximos = np.max(matrix, axis=0)
valores_minimos = np.min(matrix, axis=0)
for i in range(num_atributos):
    for j in range(num_entradas):
        matrix[j][i] = (matrix[j][i]-valores_minimos[i])/(valores_maximos[i]-valores_minimos[i])

print(matrix)

print("Introduzca el valor del learning rate")

learning_rate = float(input())

print("Introduzca el número máximo de ciclos de aprendizaje")

max_cycles = int(input())

adaline = Adaline(num_atributos, learning_rate)

cycle_count = 0

error_record = []

while cycle_count<max_cycles:

    error_absoluto_acumulado = 0
    error_absoluto_medio = 0


    for i in range(num_entradas):
        adaline.calculoSalida(matrix[i])
        error_absoluto_acumulado += error_absoluto(vector_salidas_esperadas[i], adaline.salida)
        adaline.ajustePesos(matrix[i], vector_salidas_esperadas[i])

    error_absoluto_medio = error_absoluto_acumulado / num_entradas
    error_record.append(error_absoluto_medio)
    print(error_absoluto_medio)
    cycle_count += 1

plt.plot(list(range(max_cycles)), error_record)
plt.show()









