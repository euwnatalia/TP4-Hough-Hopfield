import numpy as np

class RedHopfield:
    def __init__(self, tamano):
        self.tamano = tamano
        self.pesos = np.zeros((tamano, tamano))
    
    def entrenar(self, patrones):
        """
        Entrena la red Hopfield utilizando los patrones proporcionados.
        
        Parámetros:
        patrones (list of np.array): Lista de patrones binarios para el entrenamiento.
        
        Retorna:
        None
        """
        num_patrones = len(patrones)
        for patron in patrones:
            patron_bipolar = self._binario_a_bipolar(patron)
            self.pesos += np.outer(patron_bipolar, patron_bipolar)
        self.pesos /= num_patrones
        np.fill_diagonal(self.pesos, 0)
    
    def predecir(self, patron, pasos=10):
        """
        Predice el patrón más cercano a partir de un patrón ruidoso.
        
        Parámetros:
        patron (np.array): Patrón binario ruidoso.
        pasos (int): Número máximo de iteraciones para la convergencia.
        
        Retorna:
        np.array: Patrón binario predicho.
        """
        patron_bipolar = self._binario_a_bipolar(patron)
        for _ in range(pasos):
            patron_actualizado = np.sign(self.pesos @ patron_bipolar)
            patron_actualizado[patron_actualizado == 0] = 1
            if np.array_equal(patron_actualizado, patron_bipolar):
                break
            patron_bipolar = patron_actualizado
        return self._bipolar_a_binario(patron_bipolar)
    
    def _binario_a_bipolar(self, patron_binario):
        return np.where(np.array(list(patron_binario), dtype=int) == 0, -1, 1)

    def _bipolar_a_binario(self, patron_bipolar):
        return ''.join(['1' if x == 1 else '0' for x in patron_bipolar])
