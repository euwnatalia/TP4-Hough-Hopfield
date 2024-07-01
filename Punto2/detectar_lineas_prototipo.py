import cv2
import numpy as np

def detectar_lineas_prototipo(img_path):
    """
    Detecta líneas en una imagen utilizando la transformada de Hough.
    
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
    
    # Detectar bordes usando el detector de bordes de Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Aplicar la Transformada de Hough para detectar líneas
    lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
    
    # Dibujar las líneas detectadas en la imagen original
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    # Guardar la imagen con las líneas detectadas
    cv2.imwrite('lineas_detectadas.jpg', img)

# Ruta de la imagen
img_path = 'imagen_prueba.jpg'
detectar_lineas_prototipo(img_path)
