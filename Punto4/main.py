import numpy as np
import matplotlib.pyplot as plt
from RedHopfield import RedHopfield

def mostrar_patron(patron, titulo, tamano=10):
    """
    Muestra un patrón en una cuadrícula de 10x10.
    
    Parámetros:
    patron (np.array): Patrón binario.
    titulo (str): Título de la figura.
    tamano (int): Tamaño de la cuadrícula.
    
    Retorna:
    None
    """
    matriz_patron = np.array(list(patron), dtype=int).reshape((tamano, tamano))
    plt.imshow(matriz_patron, cmap='binary')
    plt.title(titulo)
    plt.axis('off')
    plt.show()

def mutar_patron(patron, tasa_mutacion):
    """
    Introduce un porcentaje de mutaciones en un patrón binario.
    
    Parámetros:
    patron (np.array): Patrón binario original.
    tasa_mutacion (float): Nivel de ruido (0-1) que indica el porcentaje de bits a mutar.
    
    Retorna:
    np.array: Patrón binario con ruido.
    """
    lista_patron = list(patron)
    num_mutaciones = int(len(patron) * tasa_mutacion)
    indices_a_mutar = np.random.choice(len(patron), num_mutaciones, replace=False)
    for indice in indices_a_mutar:
        lista_patron[indice] = '1' if lista_patron[indice] == '0' else '0'
    return ''.join(lista_patron)

def main():
    # Definición de patrones (caracteres binarios 10x10)
    patrones = [
        '0110100010' * 10,
        '1001001100' * 10,
        '1110101110' * 10
    ]

    # Inicializar y entrenar la red
    hopfield = RedHopfield(tamano=100)
    hopfield.entrenar(patrones)

    for patron in patrones:
        mutado = mutar_patron(patron, 0.3)
        recuperado = hopfield.predecir(mutado)
        
        print("\nPatrón Original:")
        mostrar_patron(patron, "Patrón Original")
        print("\nPatrón Mutado:")
        mostrar_patron(mutado, "Patrón Mutado")
        print("\nPatrón Recuperado:")
        mostrar_patron(recuperado, "Patrón Recuperado")

if __name__ == "__main__":
    main()
