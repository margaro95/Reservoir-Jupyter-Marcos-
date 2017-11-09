
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


def crossdata(archivo, pacient_test, segmentos=40):
    data = np.roll(archivo, (41 - pacient_test) * segmentos, axis=1)
    return data


def crossdata3(archivo, pacient_test, segmentos=40):
    data = np.roll(archivo, (41 - 14 - pacient_test) * segmentos, axis=1)
    return data


def crossvalidate(pacientes=42):
    """
    Esta función se dedica a hacer la cross validación de los pacientes para la
    clasificación en tres clases.
    """
    from reservoirclasses import Network
    #  Va a contar cuántos segmentos han sido predichos como sanos
    predictions = np.array([compute_network(Network(i)).Y for i in range(pacientes)])
    healthycomb = np.array([np.dot(np.array([[1, -1, -1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    epilepticomb = np.array([np.dot(np.array([[-1, 1, -1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    epilepticomb2 = np.array([np.dot(np.array([[-1, -1, 1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    return predictions, healthycomb, epilepticomb, epilepticomb2


def crossvalidate2(pacientes=42):
    """
    Esta función se dedica a hacer la cross validación de los pacientes para la
    clasificación en dos clases, sanos o epilepticos.
    """
    from reservoirclasses import Network
    #  Va a contar cuántos segmentos han sido predichos como sanos
    predictions = np.array([compute_network(Network(i)).Y for i in range(pacientes)])
    healthycomb = np.array([np.dot(np.array([[1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    epilepticomb = np.array([np.dot(np.array([[-1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    return predictions, healthycomb, epilepticomb


def crossvalidate3(pacientes=42-14):
    """
    Esta función se dedica a hacer la cross validación de los pacientes para la
    clasificación en dos clases, epilepticos focalizados o generales.
    Para hacer uso de esta función se debe haber creado el reservorio mediante
    Network(trainLen=1680-40-40*14) para quitar las muestras de los sanos.
    """
    from reservoirclasses import Network
    #  Va a contar cuántos segmentos han sido predichos como sanos
    predictions = np.array([compute_network(Network(i, trainLen=1680-40-40*14)).Y for i in range(pacientes)])
    generalizedcomb = np.array([np.dot(np.array([[1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    focalizedcomb = np.array([np.dot(np.array([[-1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    return predictions, generalizedcomb, focalizedcomb


def confusion_matrix(healthycomb, epilepticomb, epilepticomb2, threshold=0.8):
    healthycount = np.array([np.count_nonzero(healthycomb[i, 0, :] > threshold) for i in range(42)])
    epilepticount = np.array([np.count_nonzero(epilepticomb[i, 0, :] > threshold) for i in range(42)])
    epilepticount2 = np.array([np.count_nonzero(epilepticomb2[i, 0, :] > threshold) for i in range(42)])
    confusion_arr = np.array([[np.count_nonzero(healthycount[0:14] >= 20), np.count_nonzero(epilepticount[0:14] > 20), np.count_nonzero(epilepticount2[0:14] > 20)],
                             [[np.count_nonzero(healthycount[14:28] > 20), np.count_nonzero(epilepticount[14:28] >= 20), np.count_nonzero(epilepticount2[14:28] > 20)]],
                             [[np.count_nonzero(healthycount[28:42] > 20), np.count_nonzero(epilepticount[28:42] > 20), np.count_nonzero(epilepticount2[28:42] >= 20)]]]
                             )
    return confusion_arr
    # norm_conf = []
    # for i in confusion_arr:
        # a = 0
        # tmp_arr = []
        # a = sum(i, 0)
        # for j in i:
            # tmp_arr.append(float(j)/float(a))
        # norm_conf.append(tmp_arr)
#
    # fig = plt.figure()
    # plt.clf()
    # ax = fig.add_subplot(111)
    # ax.set_aspect(1)
    # res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet,
                    # interpolation='nearest')
#
    # width, height = confusion_arr.shape
#
    # for x in xrange(width):
        # for y in xrange(height):
            # ax.annotate(str(confusion_arr[x][y]), xy=(y, x),
                        # horizontalalignment='center',
                        # verticalalignment='center')
#
    # cb = fig.colorbar(res)
    # alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # plt.xticks(range(width), alphabet[:width])
    # plt.yticks(range(height), alphabet[:height])
    # plt.show()


def confusion_matrix2(healthycomb, epilepticomb, threshold=0.8):
    healthycount = np.array([np.count_nonzero(healthycomb[i, 0, :] > threshold) for i in range(42)])
    epilepticount = np.array([np.count_nonzero(epilepticomb[i, 0, :] > threshold) for i in range(42)])
    confusion_arr = np.array([[np.count_nonzero(healthycount[0:14] >= 20), np.count_nonzero(epilepticount[0:14] > 20)],
                             [[np.count_nonzero(healthycount[14:42] > 20), np.count_nonzero(epilepticount[14:42] >= 20)]]]
                             )
    return confusion_arr


def confusion_matrix3(generalizedcomb, focalizedcomb, threshold=0.8):
    generalizedcount = np.array([np.count_nonzero(generalizedcomb[i, 0, :] > threshold) for i in range(42 - 14)])
    focalizedcount = np.array([np.count_nonzero(focalizedcomb[i, 0, :] > threshold) for i in range(42 - 14)])
    confusion_arr = np.array([[np.count_nonzero(generalizedcount[0:14] >= 20), np.count_nonzero(focalizedcount[0:14] > 20)],
                             [[np.count_nonzero(generalizedcount[14:28] > 20), np.count_nonzero(focalizedcount[14:28] >= 20)]]]
                             )
    return confusion_arr


def readdata(archivo):
    data = io.loadmat(archivo)
    return data


def set_seed(seed=None):
    """La Seed cambia si se especifica None"""
    if seed is None:
        from time import time
        seed = int((time() * 10 ** 6) % 4294967295)
        print("Seed puesta a {}".format(seed))
    try:
        np.random.seed(seed)
        print("Seed usada:", seed)
    except TypeError:
        print("Seed no puesta correctamente")
    return seed


def initialization(nw):
    # Weights
    nw.Win = (np.random.rand(nw.resSize, 1 + nw.inSize)-0.5) * nw.input_scaling
    nw.W = np.random.rand(nw.resSize, nw.resSize)-0.5
    # Matrices
    # Allocated memory for the design (collected states) matrix
    nw.X = np.zeros((1 + nw.inSize + nw.resSize, nw.trainLen - nw.initLen))
    # Set the corresponding target matrix directly
    nw.Ytarget = nw.target[:, nw.initLen:nw.trainLen]
    # Run the reservoir with the data and collect X
    nw.x = np.zeros((nw.resSize, 1))
    return(nw)


def compute_spectral_radius(nw):

    rhoW = max(abs(linalg.eig(nw.W)[0]))
    nw.W *= nw.spectral_radius / rhoW
    return(nw)


def learning_phase(nw):
    for t in range(nw.trainLen):
        nw.u = nw.data[:, t][np.newaxis].T
        nw.x = (1-nw.a)*nw.x + nw.a*np.sin(np.dot(nw.Win, np.vstack((1, nw.u)))
                                           + np.dot(nw.W, nw.x) + 2.1)**2
        if t >= nw.initLen:
            nw.X[:, t-nw.initLen] = np.vstack((1, nw.u, nw.x))[:, 0]
    return(nw)


def train_output(nw):
    nw.X_T = nw.X.T
    if nw.reg is not None:

        nw.Wout = np.dot(
                         np.dot(nw.Ytarget, nw.X_T),
                         linalg.inv(np.dot(nw.X, nw.X_T)
                                    + nw.reg*np.eye(1 + nw.inSize + nw.resSize)
                                    )
                        )
    else:
        nw.Wout = np.dot(nw.Ytarget, linalg.pinv(nw.X))
    return(nw)


def test(nw):
    nw.Y = np.zeros((nw.outSize, nw.testLen))[np.newaxis].T
    for t in range(nw.testLen):
        nw.u = nw.data[:, nw.trainLen + t][np.newaxis].T
        nw.x = (1 - nw.a) * nw.x + nw.a * np.sin(
            np.dot(nw.Win, np.vstack((1, nw.u))) + 2.1 + np.dot(nw.W, nw.x))**2

        nw.Y[t] = np.dot(nw.Wout, np.vstack((1, nw.u, nw.x)))
    return(nw)


def compute_network(nw):
    nw = initialization(nw)
    nw = compute_spectral_radius(nw)
    nw = learning_phase(nw)
    nw = train_output(nw)
    nw = test(nw)
    return(nw)
