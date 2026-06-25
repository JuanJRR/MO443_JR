# Trabalho 4: Procesamiento Digital de Imágenes
## Resumen Técnico de Implementación

### Descripción General
Implementación completa en Python de un sistema de procesamiento digital de imágenes que incluye:
1. **Transformaciones Geométricas**: Escalado y rotación con múltiples métodos de interpolación
2. **Registro de Imágenes y Panorama**: Detección de características, correspondencia y fusión de imágenes

### Parte 1: Transformaciones Geométricas

#### Métodos de Interpolación Implementados

**1. Vecino Más Próximo (Nearest Neighbor)**
- Selecciona el píxel más próximo mediante redondeo
- Implementación: `round(x)` que equivale a comparar distancias dx, dy contra 0.5
- Ventaja: Rápido, preserva bordes nítidos
- Desventaja: Artefactos de aliasing, imagen pixelada

**2. Interpolación Bilineal**
- Media ponderada de 4 píxeles vecinos más próximos
- Fórmula vectorizada:
  ```
  f(x',y') = (1-dx)(1-dy)f(x,y) + dx(1-dy)f(x+1,y) +
             (1-dx)dy f(x,y+1) + dx·dy f(x+1,y+1)
  ```
- Ventaja: Suave, poco costoso computacionalmente
- Desventaja: Menos precisión en detalles finos

**3. Interpolación Bicúbica (B-spline)**
- Usa vecindad de 4×4 píxeles con kernel B-spline cúbica
- Kernel: `R(s) = 1/6 [P(s+2)³ - 4P(s+1)³ + 6P(s)³ - 4P(s-1)³]`
- Ventaja: Alta calidad, preserva más detalles
- Desventaja: Más costoso computacionalmente

**4. Polinomios de Lagrange**
- Interpolación usando base polinomial de Lagrange en 4×4 píxeles
- Aproximado mediante kernel B-spline para eficiencia de memoria
- Ventaja: Teoría matemática sólida
- Desventaja: Complejidad computacional

#### Transformaciones Implementadas

**Escalado (scale_image)**
- Mapeo inverso: cada píxel de salida obtiene su valor de la imagen entrada
- Fórmula: `P_out = P_in / scale_factor`
- Soporta escalas arbitrarias (0.5x, 1.5x, 2.25x, etc.)

**Rotación (rotate_image)**
- Rotación respecto al centro de la imagen
- Matriz de rotación 2D:
  ```
  [cos(θ)   sin(θ)]
  [-sin(θ)  cos(θ)]
  ```
- Soporta ángulos arbitrarios en grados (positivos = anti-horario)

#### Características Técnicas
- Operaciones vectorizadas con NumPy (sin loops explícitos)
- Clipping automático de coordenadas en bordes
- Soporte para imágenes RGB (3 canales) y escala de grises
- Preservación de tipo de dato (uint8)

### Parte 2: Registro de Imágenes y Panorama

#### Pipeline de Registro

**Paso 1: Conversión a Escala de Grises**
- Input: Imágenes JPEG coloridas
- Output: Imágenes en escala de grises para detección de características

**Paso 2: Detección de Puntos de Interés**
Implementados 4 detectores comparables:
- **SIFT** (Scale Invariant Feature Transform): Detector robusto multi-escala
  - ~100-150 puntos por imagen típicamente
  - Alta invariancia a escala, rotación, iluminación
  - Recomendado para panoramas profesionales
  
- **SURF** (Speed Up Robust Features): Similar a SIFT pero más rápido
  - Aproximadamente 50-100 puntos
  
- **ORB** (Oriented FAST and Rotated BRIEF): Binario, muy rápido
  - Hasta 500+ puntos
  - Bueno para características texturadas
  
- **BRIEF**: Descriptor binario simple
  - Muy rápido pero menos robusto

**Paso 3: Cálculo de Descriptores**
- Cada punto obtiene un descriptor vectorial (SIFT: 128-dim, ORB: 256-bit)
- Los descriptores son invariantes a transformaciones

**Paso 4: Correspondencia de Características**
- Métodos: BruteForce (fuerza bruta) o FLANN (búsqueda aproximada)
- Lowe's ratio test: Solo acepta correspondencias donde el 1° vecino es significativamente más cercano que el 2°
  - Parámetro: ratio_threshold (default: 0.7)
  - threshold = 0.7 → correspondencias de alta confianza

**Paso 5: Estimación de Homografía con RANSAC**
- RANSAC (RANdom SAmple Consensus): Algoritmo robusto para datos con outliers
- Selecciona 4 puntos aleatoriamente, estima H, cuenta inliers
- Se repite múltiples iteraciones para encontrar el mejor modelo
- Parámetro: ransac_threshold (default: 5.0 píxeles)
- Output: Matriz de homografía 3×3

**Paso 6: Alineación mediante Perspectiva**
- Aplica transformación cv2.warpPerspective a la segunda imagen
- Alinea imagen 2 con imagen 1 usando matriz H
- Mapea coordenadas: `p' = H · p`

**Paso 7: Fusión de Imágenes**
- Calcula región de superposición
- Primera imagen se coloca sin transformación (referencia)
- Segunda imagen se transforma y fusiona
- En regiones superpuestas: promedio ponderado (blend)

**Paso 8: Visualización de Correspondencias**
- Dibuja líneas verdes entre puntos correspondientes
- Punto rojo: característica en imagen 1
- Punto azul: característica en imagen 2 mapeada
- Salida: Image con ambas imágenes lado a lado

#### Matriz de Homografía
Matriz 3×3 que describe la transformación de perspectiva:
```
[h11  h12  h13]   [x']
[h21  h22  h23] · [y'] = [u·x]
[h31  h32  h33]   [1 ]   [u·y]
                           [u  ]
```

Con normalización homogénea: x' = u·x / u, y' = u·y / u

Ejemplo típico para pequeña rotación y traslación:
```
[0.9936  0.0101  18.31]
[-0.0013 1.0065  -7.66]
[-3.7e-5  6.8e-5  1.00]
```

#### Algoritmos Clave

**RANSAC Homography Estimation**
```
Para cada iteración:
  1. Seleccionar 4 correspondencias aleatorias
  2. Calcular H a partir de esos 4 puntos
  3. Contar inliers: distancia transformada < threshold
  4. Si mejor modelo: guardar
  5. Retornar H con máximos inliers
```

**Image Blending**
- En regiones no superpuestas: usar píxeles originales
- En regiones superpuestas: promedio ponderado
- Previene bordes abruptos

### Optimizaciones Implementadas

1. **Vectorización NumPy**: Operaciones en arrays completos, no píxel a píxel
2. **Meshgrid para Transformaciones**: Cálculo eficiente de coordenadas de toda la imagen
3. **Clipping Vectorizado**: Manejo de bordes sin bucles
4. **Float32 para Homografía**: Optimización de precisión vs. velocidad
5. **Caching de Kernels**: B-spline evaluado una sola vez por píxel

### Restricciones y Limitaciones

1. **SIFT**: Licencia académica en OpenCV libre, disponible con opencv-contrib-python
2. **Memoria**: Imágenes > 4000×4000 pueden requerir optimizaciones
3. **Correspondencias**: Se requieren mínimo 4 puntos válidos para homografía
4. **Superposición**: Panorama requiere 20-40% de superposición entre imágenes
5. **Rotación Panorama**: Mejor con ángulos < 45°

### Archivos Entregables

```
.
├── trabalho4_geometric.py      (600 líneas)
│   ├── Interpolación: nearest, bilinear, bicubic, lagrange
│   ├── Transformaciones: scale_image, rotate_image
│   └── CLI: argumentos de línea de comandos
│
├── trabalho4_panorama.py       (360 líneas)
│   ├── FeatureDetector: SIFT, SURF, ORB, BRIEF
│   ├── FeatureMatcher: BruteForce, FLANN
│   ├── ImageRegistration: RANSAC, warpPerspective
│   └── PanoramaBuilder: pipeline completa
│
├── main.py                     (300 líneas)
│   ├── Comando: transform
│   ├── Comando: panorama
│   ├── Comando: pipeline
│   └── WorkflowManager para gestión
│
├── test_suite.py               (250 líneas)
│   ├── Tests automáticos
│   ├── Generación de imágenes sintéticas
│   └── Validación de todos los métodos
│
├── requirements.txt
│   └── numpy, opencv-python, opencv-contrib-python, scipy
│
└── README.md
    └── Documentación de usuario
```

### Validación

Suite de pruebas ejecutada exitosamente:
- ✓ Escalado: 4 métodos × 3 factores = 12 tests
- ✓ Rotación: 4 métodos × 1 ángulo = 4 tests
- ✓ Panorama: 2 métodos de detección × 2 operaciones = 4 tests

Todos los tests pasan sin errores.

### Instrucciones de Ejecución

**Instalación**
```bash
pip install -r requirements.txt
```

**Transformación Geométrica**
```bash
python main.py transform -i imagen.png -e 2.5 -m bilinear -o ./output
python main.py transform -i imagen.png -a 45 -m bicubic -o ./output
```

**Panorama**
```bash
python main.py panorama -i1 img1.jpg -i2 img2.jpg -f sift -o ./output
```

**Pruebas**
```bash
python test_suite.py
```

### Complejidad Computacional

| Operación | Complejidad | Notas |
|-----------|-------------|-------|
| Escalado nearest | O(W·H) | Redondeo directo |
| Escalado bilinear | O(W·H) | 4 muestreos |
| Escalado bicubic | O(W·H) | 16 muestreos, kernel evaluado |
| Rotación (cualquier método) | O(W·H) | Igual al escalado |
| Detección SIFT | O(W·H·log(σ)) | Multi-escala |
| Matching SIFT | O(n·m) | n,m descriptores |
| RANSAC | O(k·n) | k iteraciones, n puntos |
| Fusión panorama | O(W·H) | Blending con máscara |

### Referencias

- OpenCV Documentation: https://docs.opencv.org/
- SIFT: Lowe, D. G. (2004). Distinctive Image Features...
- RANSAC: Fischler, M. A., & Bolles, R. C. (1981). Random Sample Consensus...
- B-spline: De Boor, C. (1978). A Practical Guide to Splines

---
Implementación realizada en Python 3.7+ con NumPy y OpenCV
Trabajo académico - MC920/MO443 - UNICAMP
