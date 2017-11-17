
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
from sklearn import linear_model
archivo = "data.mat"


def crossdata(archivo, pacient_test, segmentos=40):
    data = np.roll(archivo, (41 - pacient_test) * segmentos, axis=1)
    return data


def crossdata3(archivo, pacient_test, segmentos=40):
    data = np.roll(archivo, (41 - 14 - pacient_test) * segmentos, axis=1)
    return data


def crossvalidate(regularization, pacientes=42):
    """
    Esta función se dedica a hacer la cross validación de los pacientes para la
    clasificación en tres clases.
    """
    from reservoirclasses import Network
    #  regularization = None  # 1e-8 # "logistic"
    #  Va a contar cuántos segmentos han sido predichos como sanos
    predictions = np.array([compute_network(Network(i, outSize=3, target=crossdata(np.matrix.transpose(readdata(archivo)['targets'])[:, :], i), reg=regularization, data=crossdata(readdata(archivo)['inputs'], i)), regularization).Y for i in range(pacientes)])
    if predictions[0].shape == (3,):
        healthycomb = epilepticomb = epilepticomb2 = np.array([predictions[i, :].T[np.newaxis] for i in range(pacientes)])
        return predictions, healthycomb, epilepticomb, epilepticomb2, regularization
    try:
        healthycomb = np.array([np.dot(np.array([[1, -1, -1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
        epilepticomb = np.array([np.dot(np.array([[-1, 1, -1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
        epilepticomb2 = np.array([np.dot(np.array([[-1, -1, 1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    except:
        healthycomb = np.array([np.dot(np.array([[1, -1, -1]]), predictions[i, :, :].T) for i in range(pacientes)])
        epilepticomb = np.array([np.dot(np.array([[-1, 1, -1]]), predictions[i, :, :].T) for i in range(pacientes)])
        epilepticomb2 = np.array([np.dot(np.array([[-1, -1, 1]]), predictions[i, :, :].T) for i in range(pacientes)])
    return predictions, healthycomb, epilepticomb, epilepticomb2, regularization


def crossvalidate2(regularization, pacientes=42):
    """
    Esta función se dedica a hacer la cross validación de los pacientes para la
    clasificación en dos clases, sanos o epilepticos.
    """
    from reservoirclasses import Network
    #  regularization = "logistic"  # None # 1e-8
    #  Va a contar cuántos segmentos han sido predichos como sanos
    predictions = np.array([compute_network(Network(i, outSize=1, target=crossdata(np.matrix.transpose(readdata(archivo)['targets'])[0, :][np.newaxis], i), reg=regularization, data=crossdata(readdata(archivo)['inputs'], i)),regularization).Y for i in range(pacientes)])
    if predictions[0].shape == (2,):
        healthycomb = np.array([np.dot(np.array([[1]]), predictions[i, :].T[np.newaxis]) for i in range(pacientes)])
        epilepticomb = np.array([np.dot(np.array([[-1]]), predictions[i, :].T[np.newaxis]) for i in range(pacientes)])
        return predictions, healthycomb, epilepticomb, regularization
    try:
        healthycomb = np.array([np.dot(np.array([[1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
        epilepticomb = np.array([np.dot(np.array([[-1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    except:
        healthycomb = np.array([np.dot(np.array([[1]]), predictions[i, :].T) for i in range(pacientes)])
        epilepticomb = np.array([np.dot(np.array([[-1]]), predictions[i, :].T) for i in range(pacientes)])
    return predictions, healthycomb, epilepticomb, regularization


def crossvalidate3(regularization, pacientes=42-14):
    """
    Esta función se dedica a hacer la cross validación de los pacientes para la
    clasificación en dos clases, epilepticos focalizados o generales.
    Para hacer uso de esta función se debe haber creado el reservorio mediante
    Network2(trainLen=1680-40-40*14) para quitar las muestras de los sanos.
    """
    from reservoirclasses import Network
    #  regularization = None  # 1e-8 # "logistic"
    #  Va a contar cuántos segmentos han sido predichos como sanos
    predictions = np.array([compute_network(Network(i, outSize=1, target=crossdata3(np.matrix.transpose(readdata(archivo)['targets'])[1, 14*40:][np.newaxis], i), trainLen=1680-40-40*14, reg=regularization, data=crossdata(readdata(archivo)['inputs'][:, 14*40:], i)),regularization).Y for i in range(pacientes)])
    if predictions[0].shape == (2,):
        generalizedcomb = np.array([np.dot(np.array([[1]]), predictions[i, :].T[np.newaxis]) for i in range(pacientes)])
        focalizedcomb = np.array([np.dot(np.array([[-1]]), predictions[i, :].T[np.newaxis]) for i in range(pacientes)])
        return predictions, generalizedcomb, focalizedcomb, regularization
    try:
        generalizedcomb = np.array([np.dot(np.array([[1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
        focalizedcomb = np.array([np.dot(np.array([[-1]]), predictions[i, :, :, 0].T) for i in range(pacientes)])
    except:
        generalizedcomb = np.array([np.dot(np.array([[1]]), predictions[i, :].T) for i in range(pacientes)])
        focalizedcomb = np.array([np.dot(np.array([[-1]]), predictions[i, :].T) for i in range(pacientes)])
    return predictions, generalizedcomb, focalizedcomb, regularization


def confusion_matrix(healthycomb, epilepticomb, epilepticomb2, regularization, predictions, threshold=0):
    if regularization == "logistic":
        # healthycount = np.array([np.count_nonzero(healthycomb[i, 0, :] == 1) for i in range(42)])
        # epilepticount = np.array([np.count_nonzero(epilepticomb[i, 0, :] == 2) for i in range(42)])
        # epilepticount2 = np.array([np.count_nonzero(epilepticomb2[i, 0, :] == 3) for i in range(42)])
        # confusion_arr = np.array([[np.count_nonzero((healthycount[0:14] >= 0)&(healthycount[0:14]>epilepticount[0:14])&(healthycount[0:14]>epilepticount2[0:14])), np.count_nonzero((epilepticount[0:14] > 0)&(healthycount[0:14]<epilepticount[0:14])&(epilepticount[0:14]>epilepticount2[0:14])), np.count_nonzero((epilepticount2[0:14] > 0)&(epilepticount2[0:14]>epilepticount[0:14])&(healthycount[0:14]<epilepticount2[0:14]))],
                                 # [[np.count_nonzero((healthycount[14:28] > 0)&(healthycount[14:28]>epilepticount[14:28])&(healthycount[14:28]>epilepticount2[14:28])), np.count_nonzero((epilepticount[14:28] >= 0)&(healthycount[14:28]<epilepticount[14:28])&(epilepticount[14:28]>epilepticount2[14:28])), np.count_nonzero((epilepticount2[14:28] > 0)&(epilepticount2[14:28]>epilepticount[14:28])&(healthycount[14:28]<epilepticount2[14:28]))]],
                                 # [[np.count_nonzero((healthycount[28:42] > 0)&(healthycount[28:42]>epilepticount[28:42])&(healthycount[28:42]>epilepticount2[28:42])), np.count_nonzero((epilepticount[28:42] > 0)&(healthycount[28:42]<epilepticount[28:42])&(epilepticount[28:42]>epilepticount2[28:42])), np.count_nonzero((epilepticount2[28:42] >= 0)&(epilepticount2[28:42]>epilepticount[28:42])&(healthycount[28:42]<epilepticount2[28:42]))]]]
                                 # )
        confusion_arr = np.array([[np.count_nonzero(predictions[:14,:][np.newaxis].argmax(axis=2)==0),np.count_nonzero(predictions[:14,:][np.newaxis].argmax(axis=2)==1),np.count_nonzero(predictions[:14,:][np.newaxis].argmax(axis=2)==2)],
                                 [np.count_nonzero(predictions[14:28,:][np.newaxis].argmax(axis=2)==0),np.count_nonzero(predictions[14:28,:][np.newaxis].argmax(axis=2)==1),np.count_nonzero(predictions[14:28,:][np.newaxis].argmax(axis=2)==2)],
                                 [np.count_nonzero(predictions[28:42,:][np.newaxis].argmax(axis=2)==0),np.count_nonzero(predictions[28:42,:][np.newaxis].argmax(axis=2)==1),np.count_nonzero(predictions[28:42,:][np.newaxis].argmax(axis=2)==2)]]
                                 )
        return confusion_arr
    # healthycount = np.array([np.count_nonzero(healthycomb[i, 0, :] > threshold) for i in range(42)])
    # epilepticount = np.array([np.count_nonzero(epilepticomb[i, 0, :] > threshold) for i in range(42)])
    # epilepticount2 = np.array([np.count_nonzero(epilepticomb2[i, 0, :] > threshold) for i in range(42)])
    comb=np.zeros((42,3))
    comb[:,0]=np.mean(healthycomb[:,0,:],axis=1)
    comb[:,1]=np.mean(epilepticomb[:,0,:],axis=1)
    comb[:,2]=np.mean(epilepticomb2[:,0,:],axis=1)
    confusion_arr = np.array([[np.count_nonzero(comb[:14,:].argmax(axis=1)==0),np.count_nonzero(comb[:14,:].argmax(axis=1)==1),np.count_nonzero(comb[:14,:].argmax(axis=1)==2)],
                             [[np.count_nonzero(comb[14:28,:].argmax(axis=1)==0),np.count_nonzero(comb[14:28,:].argmax(axis=1)==1),np.count_nonzero(comb[14:28,:].argmax(axis=1)==2)]],
                             [[np.count_nonzero(comb[28:42,:].argmax(axis=1)==0),np.count_nonzero(comb[28:42,:].argmax(axis=1)==1),np.count_nonzero(comb[28:42,:].argmax(axis=1)==2)]]]
                             )
    return confusion_arr


def confusion_matrix2(healthycomb, epilepticomb, regularization, predictions, threshold=0):
    if regularization == "logistic":
        # healthycount = np.array([np.count_nonzero(healthycomb[i, 0, :] == 1) for i in range(42)])
        # epilepticount = np.array([np.count_nonzero(epilepticomb[i, 0, :] == 1) for i in range(42)])
        # confusion_arr = np.array([[np.count_nonzero((healthycount[0:14] >= 5)&(healthycount[0:14]>epilepticount[0:14])), np.count_nonzero((epilepticount[0:14] > 5)&(healthycount[0:14]<epilepticount[0:14]))],
                                 # [[np.count_nonzero((healthycount[14:42] > 5)&(healthycount[14:42]>epilepticount[14:42])), np.count_nonzero((epilepticount[14:42] >= 5)&(healthycount[14:42]<epilepticount[14:42]))]]]
                                 # )
        confusion_arr = np.array([[np.count_nonzero(predictions[:14,:][np.newaxis].argmax(axis=2)==1),np.count_nonzero(predictions[:14,:][np.newaxis].argmax(axis=2)==0)],
                                 [np.count_nonzero(predictions[14:42,:][np.newaxis].argmax(axis=2)==1),np.count_nonzero(predictions[14:42,:][np.newaxis].argmax(axis=2)==0)]]
                                 )
        return confusion_arr
    # healthycount = np.array([np.count_nonzero(healthycomb[i, 0, :] > threshold) for i in range(42)])
    # epilepticount = np.array([np.count_nonzero(epilepticomb[i, 0, :] > threshold) for i in range(42)])
    comb=np.zeros((42,2))
    comb[:,0]=np.mean(healthycomb[:,0,:],axis=1)
    comb[:,1]=np.mean(epilepticomb[:,0,:],axis=1)
    confusion_arr = np.array([[np.count_nonzero(comb[:14,:].argmax(axis=1)==0),np.count_nonzero(comb[:14,:].argmax(axis=1)==1)],
                             [[np.count_nonzero(comb[14:42,:].argmax(axis=1)==0),np.count_nonzero(comb[14:42,:].argmax(axis=1)==1)]]]
                             )
    return confusion_arr


def confusion_matrix3(generalizedcomb, focalizedcomb, regularization, predictions, threshold=0):
    if regularization == "logistic":
        # generalizedcount = np.array([np.count_nonzero(generalizedcomb[i, 0, :] == 1) for i in range(42 - 14)])
        # focalizedcount = np.array([np.count_nonzero(focalizedcomb[i, 0, :] == 1) for i in range(42 - 14)])
        # confusion_arr = np.array([[np.count_nonzero((generalizedcount[0:14] >= 5)&(generalizedcount[0:14]>focalizedcount[0:14])), np.count_nonzero((focalizedcount[0:14] > 5)&(generalizedcount[0:14]<focalizedcount[0:14]))],
                                 # [[np.count_nonzero((generalizedcount[14:28] > 5)&(generalizedcount[14:28]>focalizedcount[14:28])), np.count_nonzero((focalizedcount[14:28] >= 5)&(generalizedcount[14:28]<focalizedcount[14:28]))]]]
                                 # )
        confusion_arr = np.array([[np.count_nonzero(predictions[:14,:][np.newaxis].argmax(axis=2)==1),np.count_nonzero(predictions[:14,:][np.newaxis].argmax(axis=2)==0)],
                                 [np.count_nonzero(predictions[14:28,:][np.newaxis].argmax(axis=2)==1),np.count_nonzero(predictions[14:28,:][np.newaxis].argmax(axis=2)==0)]]
                                 )
        return confusion_arr
    comb=np.zeros((28,2))
    comb[:,0]=np.mean(generalizedcomb[:,0,:],axis=1)
    comb[:,1]=np.mean(focalizedcomb[:,0,:],axis=1)
    confusion_arr = np.array([[np.count_nonzero(comb[:14,:].argmax(axis=1)==0),np.count_nonzero(comb[:14,:].argmax(axis=1)==1)],
                             [[np.count_nonzero(comb[14:28,:].argmax(axis=1)==0),np.count_nonzero(comb[14:28,:].argmax(axis=1)==1)]]]
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


def initialization(nw, regularization):
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
        # WATCH OUT!!!!!! IF YOU WANT TO USE LEAK RATE, YOU MUST USE THE NEXT
        # LINE WITH np.dot(nw.W, nw.x)
        nw.x = (1-nw.a)*nw.x + nw.a*np.sin(np.dot(nw.Win, np.vstack((1, nw.u)))
                                           # + np.dot(nw.W, nw.x) + 2.1)**2
                                           + 2.1)**2
        if t >= nw.initLen:
            nw.X[:, t-nw.initLen] = np.vstack((1, nw.u, nw.x))[:, 0]
    return(nw)


def train_output(nw):
    nw.X_T = nw.X.T
    if (nw.reg is not None) & (nw.reg != "logistic"):
        nw.Wout = np.dot(
                         np.dot(nw.Ytarget, nw.X_T),
                         linalg.inv(np.dot(nw.X, nw.X_T)
                                    + nw.reg*np.eye(1 + nw.inSize + nw.resSize)
                                    )
                         )
    elif nw.reg == "logistic":
        model = linear_model.LogisticRegression(dual=True, class_weight='balanced', tol=1e-7)
        nw.Wout = model.fit(nw.X_T, np.ravel(nw.Ytarget.T, order='F'))
        # np.dot(np.log(((nw.Ytarget+1)/2)/(1 - ((nw.Ytarget+1)/2))), nw.X_T)

    else:
        nw.Wout = np.dot(nw.Ytarget, linalg.pinv(nw.X))
    return(nw)


def test(nw):
    nw.Y = np.zeros((nw.outSize, nw.testLen))[np.newaxis].T
    nw.U = np.zeros((1 + np.size(nw.data, 0) + nw.resSize, nw.testLen))
    for t in range(nw.testLen):
        nw.u = nw.data[:, nw.trainLen + t][np.newaxis].T
        # WATCH OUT!!!!!! IF YOU WANT TO USE LEAK RATE, YOU MUST USE THE NEXT
        # LINE WITH np.dot(nw.W, nw.x)
        nw.x = (1 - nw.a) * nw.x + nw.a * np.sin(
            np.dot(nw.Win, np.vstack((1, nw.u))) + 2.1)**2
                                                    # + np.dot(nw.W, nw.x))**2
        if nw.reg != "logistic":
            nw.Y[t] = np.dot(nw.Wout, np.vstack((1, nw.u, nw.x)))
        nw.U[:, t] = np.vstack((1, nw.u, nw.x))[:, 0]
    if nw.reg == "logistic":
        nw.Y = np.mean(nw.Wout.predict_proba(nw.U.T), axis=0)
        print("Clases\n {}".format(nw.Wout.predict(nw.U.T)))
        print("Probabilidades\n {}".format(nw.Wout.predict_proba(nw.U.T)))

    return(nw)


def compute_network(nw, regularization):
    nw = initialization(nw, regularization)
    nw = compute_spectral_radius(nw)
    nw = learning_phase(nw)
    nw = train_output(nw)
    nw = test(nw)
    return(nw)
