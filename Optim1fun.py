from reservoirclasses import*


def Optim1fun(mode, regression, iscalin, phase, resSize, tolerancia=1e-4):
    if mode == "triple":
        [predictions, healthycomb, epilepticomb, epilepticomb2, regularization] = crossvalidate(regularization=10**(regression*1e5), tolerancia=tolerancia, phase=0.1*phase*1e5, inputscalin=10**(iscalin*1e5), resSize=100*np.abs(resSize*1e5))
        conf_arr = confusion_matrix(healthycomb, epilepticomb, epilepticomb2, regularization, predictions)
        J = 42 - np.trace(conf_arr)
        print("Función de coste en esta iteración{}".format(J))
    elif mode == "healthyvsepileptic":
        [predictions, healthycomb, epilepticomb, regularization] = crossvalidate2(regularization=10**(regression), tolerancia=tolerancia, phase=0.1*phase, inputscalin=10**(iscalin), resSize=100*resSize)
        conf_arr = confusion_matrix2(healthycomb, epilepticomb, regularization, predictions)
        J = 42 - np.trace(conf_arr)

    elif mode == "generalizedvsfocalized":
        [predictions, generalizedcomb, focalizedcomb, regularization] = crossvalidate3(regularization=10**(regression), tolerancia=tolerancia, phase=0.1*phase, inputscalin=10**(iscalin), resSize=100*resSize)
        conf_arr = confusion_matrix3(generalizedcomb, focalizedcomb, regularization, predictions)
        J = 28 - np.trace(conf_arr)

    return J
