# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:54:04 2017

Se define la clase Network, que tiene todas las propiedades del reservorio.

@author: marcos
"""
from reservoirfunctions import*
import numpy as np


class Network(object):

    def __init__(self, pacient_test, reg, data, outSize, target, resSize=500, a=1, sr=1.25, iscaling=0.1,
                 trainLen=int(1680-40), testLen=40, initLen=0):
        #  Si quieres hacer cambios en la clase para adaptarlo al tipo de previsión, tienes que cambiar
        #  tanto el self.data y el self.target como el self.outSize
        #  reg puede tomar los valores None (para pseudo-inverse), numérico (1e-8) para ridge regression y "logistic" para regresion logistica
        self.pacient_test = pacient_test
        self.initLen = initLen
        self.trainLen = trainLen
        self.testLen = testLen
        self.data = data
        # crossdata(readdata(archivo)['inputs'], pacient_test)  # Usar con crossvalidation() y crossvalidation2()
        # crossdata(readdata(archivo)['inputs'][:, 14*40:], pacient_test)  # Usar con crossvalidation3
        if (reg == "logistic") & (outSize == 3):
            self.target = crossdata(np.array(([1]*14*40 + [2]*14*40 + [3]*14*40))[np.newaxis], pacient_test)
        else:
            self.target = target
        # crossdata(np.matrix.transpose(readdata(archivo)['targets'])[0, :][np.newaxis], pacient_test)  # Caso de crossvalidation2()
        # crossdata(np.matrix.transpose(readdata(archivo)['targets'])[:, :], pacient_test)  # Caso de crossvalidation()
        # crossdata3(np.matrix.transpose(readdata(archivo)['targets'])[1, 14*40:][np.newaxis], pacient_test)  # Caso de crossvalidation3()
        self.inSize = np.size(self.data, axis=0)
        self.outSize = outSize
        # 1  # Usar con crossvalidation() y crossvalidation2()
        # 3  # Usar con crossvalidation3()

        self.resSize = resSize  # Reservoir size (prediction)
        # self.resSize = 1000 #Reservoir size (generation)
        self.a = a  # Leak rate alpha
        self.spectral_radius = sr  # 1.25  # Spectral raidus
        self.input_scaling = iscaling  # Input scaling
        self.reg = reg  # 1e-8  # None #Regularization #  "logistic" factor - if None,
        # we'd use pseudo-inverse rather than ridge regression

        # Change the seed, reservoir performances should be averaged accross
        # at least 20 random instances (with the same set of parameters)
        seed = None  # 42

        set_seed(seed)
