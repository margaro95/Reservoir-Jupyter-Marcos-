# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:54:04 2017

Se define la clase Network, que tiene todas las propiedades del reservorio.

@author: marcos
"""
from reservoirfunctions import*
import numpy as np


class Network(object):

    def __init__(self, pacient_test, archivo="data.mat", trainLen=1680-40,
                 testLen=40, initLen=0):
        self.pacient_test = pacient_test
        self.initLen = initLen
        self.trainLen = trainLen
        self.testLen = testLen
        self.data = crossdata(readdata(archivo)['inputs'][:, 14*40:], pacient_test)  # Usar con crossvalidation3
        # crossdata(readdata(archivo)['inputs'], pacient_test)  # Usar con crossvalidation() y crossvalidation2()
        self.target = crossdata3(np.matrix.transpose(readdata(archivo)['targets'])[1, 14*40:][np.newaxis], pacient_test)  # Caso de crossvalidation3()
        # crossdata(np.matrix.transpose(readdata(archivo)['targets'])[0, :][np.newaxis], pacient_test)  # Caso de crossvalidation2()
        # crossdata(np.matrix.transpose(readdata(archivo)['targets'])[:, :], pacient_test)  # Caso de crossvalidation()

        self.inSize = np.size(self.data, axis=0)
        self.outSize = 1  # Será uno para clasificación en dos y tres para tres
        self.resSize = 500  # Reservoir size (prediction)
        # self.resSize = 1000 #Reservoir size (generation)
        self.a = 0.3  # Leak rate alpha
        self.spectral_radius = 1  # 1.25  # Spectral raidus
        self.input_scaling = 0.1  # Input scaling
        self.reg = None  # 1e-8  # None #Regularization factor - if None,
        # we'd use pseudo-inverse rather than ridge regression

        # Change the seed, reservoir performances should be averaged accross
        # at least 20 random instances (with the same set of parameters)
        seed = None  # 42

        set_seed(seed)
