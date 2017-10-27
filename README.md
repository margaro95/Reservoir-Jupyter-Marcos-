# Reservoir-Jupyter-Marcos-

Implementación de un reservorio para el estudio de la *epilepsia*.

## Uso

Se requiere definir una instancia de la clase Network, dándole los datos que quieres procesar.
Después, se utilizarán sobre ella las funciones de initialization, compute_spectral_radius, learning_phase, train_output, test y compute_error.

El training de la capa de salida se puede realizar no solo mediante la pseudo-inverse sino también mediante ridge regression si utilizamos regularización en la definición de la instancia de Network.
