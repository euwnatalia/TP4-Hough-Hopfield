import cv2
import numpy as np

def detectar_circunferencias_prototipo(img_path):
    """
    Detecta circunferencias en una imagen utilizando la transformada de Hough.
    
    Parámetros:
    img_path (str): Ruta de la imagen a procesar.
    
    Retorna:
    None
    """
    # Cargar imagen y convertir a escala de grises
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en la ruta: {img_path}")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    # Aplicar la Transformada de Hough para detectar circunferencias
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=50, param2=30, minRadius=0, maxRadius=0)
    
    # Dibujar las circunferencias detectadas en la imagen original
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # Dibujar la circunferencia
            cv2.circle(img, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # Dibujar el centro de la circunferencia
            cv2.circle(img, (i[0], i[1]), 2, (0, 0, 255), 3)
    
    # Guardar la imagen con las circunferencias detectadas
    cv2.imwrite('circunferencias_detectadas.jpg', img)

# Ruta de la imagen
img_path = 'imagen_prueba.jpg'
detectar_circunferencias_prototipo(img_path)
