
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 11:54:04 2017

Se crean todas las funciones que permiten leer archivos de MATLAB, iniciar el
reservorio, entrenarlo y testearlo.

@author: marcos
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg, io
# from ipywidgets import *
# from IPython.display import *


def readdata(archivo):
    data = io.loadmat(archivo)
    return data


def set_seed(seed=None):
    """Making the seed (for random values) variable if None"""
    if seed is None:
        from time import time
        seed = int((time() * 10 ** 6) % 4294967295)
        print("Seed set to {}".format(seed))
    try:
        np.random.seed(seed)
        print("Seed used for random values:", seed)
    except TypeError:
        print("!!! WARNING !!!: Seed was not set correctly.")
    return seed


def plot_figure(segment, patient, nw):
    plt.figure(0).clear()
    plt.plot(nw.data[:, int((segment - 1) + 40 * (patient - 1))])
    plt.ylim([-0.1, 1.1])
    plt.title(
              '{0}th segment of data from {1}th patient'.format(segment,
                                                                patient)
              )


def initialization(nw):
    # Weights
    nw.Win = (np.random.rand(nw.resSize, 1 + nw.inSize)-0.5) * nw.input_scaling
    nw.W = np.random.rand(nw.resSize, nw.resSize)-0.5
    # Matrices
    # Allocated memory for the design (collected states) matrix
    nw.X = np.zeros((1 + nw.inSize + nw.resSize, nw.trainLen - nw.initLen))
    # Set the corresponding target matrix directly
    nw.Ytarget = nw.target[:, nw.initLen+1:nw.trainLen+1]
    # Run the reservoir with the data and collect X
    nw.x = np.zeros((nw.resSize, 1))
    return(nw)


def compute_spectral_radius(nw):
    print('Computing spectral radius...', end=" ")
    rhoW = max(abs(linalg.eig(nw.W)[0]))
    print('Done.')
    nw.W *= nw.spectral_radius / rhoW
    return(nw)


def learning_phase(nw):
    for t in range(nw.trainLen):
        # Input data
        nw.u = nw.data[:, t]
        nw.x = (1-nw.a)*nw.x + nw.a*(np.sin(np.dot(nw.Win, np.vstack((1, nw.u[np.newaxis].T))
                                                   ) + 2.1 + np.dot(nw.W, nw.x)))**2
        # After the initialization, we start modifying X
        if t >= nw.initLen:
            nw.X[:, t-nw.initLen] = np.vstack((1, nw.u[np.newaxis].T, nw.x))[:, 0]
    return(nw)


def train_output(nw):
    nw.X_T = nw.X.T
    if nw.reg is not None:
        # Ridge regression (linear regression with regularization)
        nw.Wout = np.dot(
                         np.dot(nw.Ytarget, nw.X_T),
                         linalg.inv(np.dot(nw.X, nw.X_T)
                                    + nw.reg*np.eye(1 + nw.inSize + nw.resSize)
                                    )
                        )
    else:
        # Pseudo-inverse
        nw.Wout = np.dot(nw.Ytarget, linalg.pinv(nw.X))
    return(nw)


def test(nw):
    # Run the trained ESN in a generative mode. no need to initialize here,
    # because x is initialized with training data and we continue from there.
    nw.Y = np.zeros((nw.outSize, nw.testLen))
    nw.u = nw.data[:, nw.trainLen]
    for t in range(nw.testLen):
        nw.x = (1 - nw.a) * nw.x + nw.a * \
                                        (np.sin(
                                                np.dot(nw.Win,
                                                       np.vstack((1, nw.u[np.newaxis].T))) +
                                                np.dot(nw.W, nw.x) + 2.1
                                                ))**2
        nw.y = np.dot(nw.Wout, np.vstack((1, nw.u[np.newaxis].T, nw.x)))
        nw.Y[:, t] = nw.y  # Esto es lo que vale, la predicción
        if nw.mode == 'generative':
            # Generative mode:
            nw.u = nw.y
        elif nw.mode == 'prediction':
            # Predictive mode:
            nw.u = nw.data[nw.trainLen + t + 1]
        else:
            raise(Exception, "ERROR: 'mode' was not set correctly.")
    return(nw)


def compute_error(nw):
    # Computing MSE for the first errorLen iterations
    errorLen = 500
    mse = sum(np.square(nw.data[nw.trainLen + 1:nw.trainLen + errorLen + 1] -
              nw.Y[0, 0:errorLen])) / errorLen
    print('MSE = ' + str(mse))
    return(nw)


def compute_network(nw):
    nw = initialization(nw)
    nw = compute_spectral_radius(nw)
    nw = learning_phase(nw)
    nw = train_output(nw)
    nw = test(nw)
    nw = compute_error(nw)
    return(nw)
