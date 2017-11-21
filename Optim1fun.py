from reservoirclasses import*


def Optim1fun(mode, regression, iscalin, phase, resSize, tolerancia=1e-4):
    if mode == "triple":
        [predictions, healthycomb, epilepticomb, epilepticomb2, regularization] = crossvalidate(regularization=regression, tolerancia=tolerancia, phase=phase, inputscalin=iscalin, resSize=resSize)
        conf_arr = confusion_matrix(healthycomb, epilepticomb, epilepticomb2, regularization, predictions)
        J = 42 - np.trace(conf_arr)

    elif mode == "healthyvsepileptic":
        [predictions, healthycomb, epilepticomb, regularization] = crossvalidate2(regularization=regression, tolerancia=tolerancia, phase=phase, inputscalin=iscalin, resSize=resSize)
        conf_arr = confusion_matrix2(healthycomb, epilepticomb, regularization, predictions)
        J = 42 - np.trace(conf_arr)

    elif mode == "generalizedvsfocalized":
        [predictions, generalizedcomb, focalizedcomb, regularization] = crossvalidate3(regularization=regression, tolerancia=tolerancia, phase=phase, inputscalin=iscalin, resSize=resSize)
        conf_arr = confusion_matrix3(generalizedcomb, focalizedcomb, regularization, predictions)
        J = 28 - np.trace(conf_arr)

    return J
