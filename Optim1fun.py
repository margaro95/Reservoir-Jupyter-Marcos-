from reservoirclasses import*


def Optim1Fun(mode, regression, tolerancia, iscalin, phase, regularizacion):
	if mode=="triple":
		[predictions, healthycomb, epilepticomb, epilepticomb2, regularization] = crossvalidate(regression, tolerancia, iscalin, phase, regularizacion)
		conf_arr = confusion_matrix(healthycomb, epilepticomb, epilepticomb2, regularization, predictions)
		J = 42 - np.trace(conf_arr)

	elif mode=="healthyvsepileptic":
		[predictions, healthycomb, epilepticomb, regularization] = crossvalidate2(regression, tolerancia, iscalin, phase, regularizacion)
		conf_arr = confusion_matrix2(healthycomb, epilepticomb, regularization, predictions)
		J = 42 - np.trace(conf_arr)

	elif mode=="generalizedvsfocalized":
		[predictions, generalizedcomb, focalizedcomb, regularization] = crossvalidate3(regression, tolerancia, iscalin, phase, regularizacion)
		conf_arr = confusion_matrix3(generalizedcomb, focalizedcomb, regularization, predictions)
		J = 28 - np.trace(conf_arr)
	
	return J