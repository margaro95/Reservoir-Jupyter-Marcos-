
# -*- coding: utf-8 -*-
"""
En este archivo se implementará la optimización del reservorio mediante algún
tipo de función del estilo fminunc o fmincon
"""

from reservoirclasses import*
import scipy.optimize
from Optim1fun import Optim1fun

Nfeval=1


def callbackF(Xi,f,accept):
    global Nfeval
    print("{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}".format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Optim1fun("triple", Xi[0], Xi[1], Xi[2], Xi[3])))
    Nfeval += 1
    input("Pulsa para continuar...")


def optimizacion():
    xopt = scipy.optimize.basinhopping(lambda x: Optim1fun("triple", x[0], x[1], x[2], x[3]), x0=[-2e-5, -1e-5, 21e-5, 3.5e-5],
                                       callback=callbackF, stepsize=5, take_step=rutina_pasos())
    Jopt = Optim1fun("triple", xopt[0], xopt[1], xopt[2], xopt[3])
    return xopt, Jopt


class rutina_pasos(object):
    def __init__(self, stepsize=5):
        self.stepsize = stepsize

    def __call__(self, x):
        s = self.stepsize
        x += np.random.uniform(-s, s, x.shape)
        return x
