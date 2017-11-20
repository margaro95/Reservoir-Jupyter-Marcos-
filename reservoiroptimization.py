
# -*- coding: utf-8 -*-

"""
En este archivo se implementará la optimización del reservorio mediante algún
tipo de función del estilo fminunc o fmincon
"""

from reservoirclasses import*
import scipy.optimize
import Optim1fun

#It's just a straight-forward conversion from Matlab syntax to python syntax:

#import scipy.optimize

#banana = lambda x: 100*(x[1]-x[0]**2)**2+(1-x[0])**2
#xopt = scipy.optimize.fmin(func=banana, x0=[-1.2,1])
#with output:

#Optimization terminated successfully.
#         Current function value: 0.000000
#         Iterations: 85
#         Function evaluations: 159
#array([ 1.00002202,  1.00004222])

def optimizacion(x)
	funcion = lambda x: Optim1fun("triple", x[0], x[1], x[2])