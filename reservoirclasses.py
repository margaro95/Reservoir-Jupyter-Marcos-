# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:54:04 2017

Se define la clase Network, que tiene todas las propiedades del reservorio.

@author: marcos
"""
from reservoirfunctions import set_seed, readdata, crossdata
import numpy as np


class Network(object):

    def __init__(self, pacient_test, archivo="data.mat", trainLen=1680-40,
                 testLen=40, initLen=0):
        self.pacient_test = pacient_test
        self.initLen = initLen
        self.trainLen = trainLen
        self.testLen = testLen
        self.data = crossdata(readdata(archivo)['inputs'], pacient_test)
        self.target = crossdata(np.matrix.transpose(
            readdata(archivo)['targets'])[:, :], pacient_test)
        self.inSize = np.size(self.data, axis=0)
        self.outSize = 3  # Sano (1) o epiléptico (-1) ó sano vs dos epilepsias
        self.resSize = 300  # Reservoir size (prediction)
        # self.resSize = 1000 #Reservoir size (generation)
        self.a = 0.3  # Leak rate alpha
        self.spectral_radius = 1.25  # Spectral raidus
        self.input_scaling = 1  # Input scaling
        self.reg = None  # 1e-8  # None #Regularization factor - if None,
        # we'd use pseudo-inverse rather than ridge regression

        # Change the seed, reservoir performances should be averaged accross
        # at least 20 random instances (with the same set of parameters)
        seed = None  # 42

        set_seed(seed)
