import numpy as np
import random


class Adaline:

    def __init__(self, celulas_entrada, learning_rate):
        self.learning_rate = learning_rate
        self.umbral = random.random()  #Valor umbral aleatorio
        # Pesos aleatorios
        self.pesos = np.zeros(celulas_entrada, dtype= float)
        for x in range(celulas_entrada):
            self.pesos[x] = random.random()
        self.salida = 0

    def ajustePesos (self, entrada, salida_esperada):

        diferencia = salida_esperada - self.salida

        #Ajuste umbral
        incremento_u = self.learning_rate*diferencia
        self.umbral += incremento_u

        #Ajuste de los pesos
        for i in range(len(self.pesos)):
            incremento_p = self.learning_rate*diferencia*entrada[i]
            self.pesos[i] += incremento_p

    def calculoSalida (self, entrada):

       sumatorio = np.dot(self.pesos, entrada)
       self.salida = sumatorio + self.umbral

