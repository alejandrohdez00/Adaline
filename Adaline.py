import numpy as np
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

