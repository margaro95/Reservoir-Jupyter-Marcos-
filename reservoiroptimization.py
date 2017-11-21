
# -*- coding: utf-8 -*-
"""
En este archivo se implementará la optimización del reservorio mediante algún
tipo de función del estilo fminunc o fmincon
"""

from reservoirclasses import*
import scipy.optimize
from Optim1fun import Optim1fun

Nfeval=1


def callbackF(Xi):
    global Nfeval
    print("{0:4d}   {1: 3.6f}   {2: 3.6f}   {3: 3.6f}   {4: 3.6f}   {5: 3.6f}".format(Nfeval, Xi[0], Xi[1], Xi[2], Xi[3], Optim1fun(Xi)))
    Nfeval += 1
    input("Press Enter to continue...")


def optimizacion():
    xopt = scipy.optimize.fmin(lambda x: Optim1fun("triple", x[0], x[1], x[2], x[3]), x0=[1e-2, 1e-1, 2.1, 500], callback=callbackF)
    Jopt = Optim1fun("triple", xopt[0], xopt[1], xopt[2], xopt[3])
    return xopt, Jopt
