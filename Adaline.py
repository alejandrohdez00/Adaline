import numpy as np
import panda as pd
import random

class Adaline:

    def __init__(self, numero_entradas, learning_rate):
        self.learning_rate = learning_rate
        self.umbral = random.random()  #Valor umbral aleatorio
        # Pesos aleatorios
        self.pesos = []
        for x in range(numero_entradas):
            self.pesos[x] = random.random()
        self.salida = 0

    def ajustarPesos(self, entradas, salida_esperada):
        #Calculamos diferencia entre salida esperada y salida
        diferencia = salida_esperada - self.salida

        #Actualizamos umbral
        incremento_u = self.learning_rate*diferencia
        self.umbral += incremento_u

        #Actualizamos pesos
        for i in range(len(self.pesos)):
            incremento_p = self.learning_rate*diferencia*entradas[i]
            self.pesos[i] += incremento_p

    def calculoSalida(self, entradas):
        #Calculamos el sumatorio de las entradas por su peso asignado
        sumatorio = 0






