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

    def AjustePesos (self, entrada, salida_esperada):
        diferencia = salida_esperada - self.salida

        #Ajuste umbral
        incremento_u = self.aprendizaje*diferencia
        self.umbral += incremento_u

        #Ajuste de los pesos
        for x in range(len(self.pesos)):
            incremento_p = self.learning_rate*diferencia*entrada[x]
            self.pesos[x] += incremento_p

    def CalculoSalida (self, entrada):
        sumatorio=0

        # Utilizar producto vectorial para la suma de pesos y entradas (numpy)

        self.salida= sumatorio + self.umbral
