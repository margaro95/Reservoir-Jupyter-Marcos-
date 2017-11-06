# TODO
Ya hemos hecho la matriz de confusión, dando como resultado lo siguiente:
array([list([11, 3, 3]), list([[1, 5, 6]]), list([[2, 5, 7]])], dtype=object)

El reservorio distingue muy bien los sanos de los epilépticos pero no lo hace
tan bien cuando tiene que distinguir entre epilepticos. Hay algo raro tambien...
La matriz de confusión devuelve 43 pacientes... Pero solo se tienen 42. Hay
algún error ahí. Hay que probar tambien en meterle input scaling para ver si
cambia la performance al hacer ridge regression en lugar de pseudo inversa.
También hay que aumentar el número de nodos para ver si esto aumenta la capacidad
de separación del reservorio.
Si nada de esto da resultado en una mejor previsión, toca hacer la clasificación
en dos pasos, como hizo Miguel.
