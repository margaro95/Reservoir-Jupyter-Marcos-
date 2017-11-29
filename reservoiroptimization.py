
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


a=np.array([1,0,0,0])
b=np.array([0,1,0,0])
c=np.array([0,0,1,0])
d=np.array([0,0,0,1])


def optimizacion():
    global a, b, c, d
    xopt = scipy.optimize.minimize(lambda x: Optim1fun("triple", x[0], x[1], x[2], x[3]), x0=[1e-3, 1e-2, 0.21, 3.5], method='Powell', callback=callbackF, options={'direc':[d,b,c,a]})
    Jopt = xopt.fun#Optim1fun("triple", xopt[0], xopt[1], xopt[2], xopt[3])
    return xopt, Jopt


#class rutina_pasos(object):
#    def __init__(self, stepsize=5):
#        # self.stepsize = stepsize
#
#    def __call__(self, x):
#        s =5
#        x += s
#        return x
