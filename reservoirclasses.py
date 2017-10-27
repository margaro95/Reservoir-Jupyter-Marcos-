# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:54:04 2017

Se define la clase Network, que tiene todas las propiedades del reservorio.

@author: marcos
"""
from reservoirfunctions import set_seed, readdata
import numpy as np


class Network(object):

    def __init__(self, trainLen=1680-40, testLen=40, initLen=0):
        self.initLen = initLen
        self.trainLen = trainLen
        self.testLen = testLen
        self.data = readdata("data.mat")['inputs']
        self.target = np.matrix.transpose(readdata("data.mat")['targets']
                                          )[[0, 1], :]
        self.inSize = np.size(self.data, axis=0)
        self.outSize = 2  # Sano (1) o epiléptico (-1)
        self.resSize = 300  # Reservoir size (prediction)
        # self.resSize = 1000 #Reservoir size (generation)
        self.a = 0.3  # Leak rate alpha
        self.spectral_radius = 1.25  # Spectral raidus
        self.input_scaling = 0.1  # Input scaling
        self.reg = 1e-8  # None #Regularization factor - if None,
        # we'd use pseudo-inverse rather than ridge regression

        self.mode = 'prediction'
        # self.mode = 'generative'

        # Change the seed, reservoir performances should be averaged accross
        # at least 20 random instances (with the same set of parameters)
        seed = None  # 42

        set_seed(seed)
