# Trabalho 4: Procesamiento Digital de Imágenes

Implementación de transformaciones geométricas e imagen panorámica en Python.

## Estructura del Proyecto

```
.
├── trabalho4_geometric.py      # Transformaciones geométricas (escala, rotación, interpolación)
├── trabalho4_panorama.py       # Registro de imágenes y construcción de panorama
├── main.py                     # Pipeline principal con interfaz CLI
├── test_suite.py               # Suite de pruebas con imágenes sintéticas
├── requirements.txt            # Dependencias de Python
└── README.md                   # Este archivo
```

## Instalación de Dependencias

```bash
pip install -r requirements.txt
```

Requisitos principales:
- Python 3.7+
- NumPy 1.21+
- OpenCV 4.5+ (opencv-python + opencv-contrib-python)
- SciPy 1.7+

## Uso

### 1. Transformaciones Geométricas

#### Escalado
```bash
python main.py transform -i imagen.png -e 2.5 -m bilinear -o ./output
```

#### Rotación
```bash
python main.py transform -i imagen.png -a 45 -m bicubic -o ./output
```

**Parámetros:**
- `-i, --input`: Ruta de imagen PNG
- `-e, --scale`: Factor de escala (float, ej: 0.5, 1.5, 2.0)
- `-a, --angle`: Ángulo de rotación en grados (float, ej: 45.5)
- `-m, --method`: Método de interpolación
  - `nearest`: Vecino más próximo
  - `bilinear`: Interpolación bilineal
  - `bicubic`: Interpolación bicúbica (B-spline)
  - `lagrange`: Polinomios de Lagrange
- `-o, --output-dir`: Directorio de salida

### 2. Construcción de Panorama

```bash
python main.py panorama -i1 imagen1.jpg -i2 imagen2.jpg -f sift -o ./output
```

**Parámetros:**
- `-i1, --input1`: Primera imagen JPEG
- `-i2, --input2`: Segunda imagen JPEG
- `-f, --feature`: Método de detección
  - `sift`: Scale Invariant Feature Transform (recomendado)
  - `surf`: Speed Up Robust Features
  - `orb`: Oriented FAST and Rotated BRIEF
  - `brief`: Binary Robust Independent Elementary Features
- `-t, --ratio-threshold`: Umbral de ratio para filtrado (default: 0.7)
- `-r, --ransac-threshold`: Umbral RANSAC para homografía (default: 5.0)
- `-o, --output-dir`: Directorio de salida

**Salidas:**
- `correspondencias.jpg`: Imágenes lado a lado con líneas de correspondencia
- `panorama.jpg`: Imagen panorámica fusionada

### 3. Pipeline Completa

```bash
python main.py pipeline -i imagen.png -e 1.5 -i1 img1.jpg -i2 img2.jpg -o ./output
```

## Pruebas

Ejecutar suite de pruebas con imágenes generadas sintéticamente:

```bash
python test_suite.py
```

Esto genera:
- Imágenes de prueba
- Tests de todos los métodos de interpolación
- Tests de escalado y rotación
- Tests de panorama con SIFT y ORB

Los resultados se guardan en `test_data/output/`

## Implementación de Algoritmos

### Transformaciones Geométricas

#### Vecino Más Próximo
Selecciona el píxel más cercano al punto interpolado.

#### Interpolación Bilineal
Media ponderada de los 4 píxeles vecinos más próximos.

#### Interpolación Bicúbica
Usa una vecindad de 4×4 píxeles con kernel B-spline cúbico:
```
R(s) = 1/6 [P(s+2)³ - 4P(s+1)³ + 6P(s)³ - 4P(s-1)³]
```

#### Polinomios de Lagrange
Interpolación de 4×4 píxeles usando base polinomial de Lagrange.

### Registro de Imágenes

1. **Detección de características**: Detecta puntos de interés invariantes a escala y rotación
2. **Descripción**: Calcula descriptores vectoriales para cada punto
3. **Correspondencia**: Empareja puntos similares entre imágenes
4. **Filtrado RANSAC**: Estima matriz de homografía H eliminando outliers
5. **Alineación**: Aplica transformación de perspectiva
6. **Fusión**: Combina imágenes alineadas en panorama

## Funciones Principales

### trabalho4_geometric.py

```python
scale_image(img, scale_factor, method='bilinear', output_size=None)
rotate_image(img, angle_deg, method='bilinear', output_size=None)
apply_transformation(input_path, output_path, angle=0, scale=1, 
                    method='bilinear', output_size=None)
```

### trabalho4_panorama.py

```python
class FeatureDetector:
    def detect_and_compute(img_gray)

class FeatureMatcher:
    @staticmethod
    def match_features(desc1, desc2, method='brute_force', ratio_threshold=0.7)

class ImageRegistration:
    @staticmethod
    def estimate_homography(src_pts, dst_pts, method='ransac', ransac_threshold=5.0)
    
    @staticmethod
    def create_panorama(img1, img2, H)

class PanoramaBuilder:
    def register_images(img1_path, img2_path, ratio_threshold=0.7, ransac_threshold=5.0)
    def build_panorama(img1, img2)
    def draw_matches(img1, img2, kp1, kp2, matches, output_path)

process_panorama(img1_path, img2_path, matches_output_path, panorama_output_path,
                feature_method='sift', matcher_method='brute_force',
                ratio_threshold=0.7, ransac_threshold=5.0)
```

## Consideraciones de Implementación

### Interpolación
- Operaciones totalmente vectorizadas con NumPy
- Manejo de bordes con clipping automático
- Soporte para imágenes RGB y escala de grises
- Preservación de tipo de dato (uint8)

### Registro de Imágenes
- Uso de OpenCV para detección robusta de características
- RANSAC para estimación robusta de homografía
- Fusión inteligente de regiones superpuestas
- Manejo de imágenes de diferentes tamaños

### Optimizaciones
- Operaciones vectorizadas en lugar de loops
- Uso de meshgrid para mapeo de coordenadas
- Cache de kernels de interpolación
- Clipping eficiente de bordes

## Limitaciones y Consideraciones

1. **SIFT no está disponible en OpenCV 4.4 libre**: Se proporciona alternativa ORB
2. **Memoria**: Imágenes muy grandes pueden requerir más memoria
3. **Correspondencias**: Se requieren al menos 4 correspondencias válidas
4. **Superposición**: Para panorama se recomienda 20-40% de superposición
5. **Ángulo de panorama**: Funciona mejor con rotaciones pequeñas (< 45°)

## Archivos de Salida

### Transformaciones Geométricas
- `transformacion_resultado.png`: Imagen transformada en PNG

### Panorama
- `correspondencias.jpg`: Imagen mostrando líneas de correspondencia
- `panorama.jpg`: Imagen panorámica final

## Ejemplos de Entrada

Se pueden descargar imágenes de ejemplo de:
- Transformaciones: http://www.ic.unicamp.br/~helio/imagens_png/
- Panorama: http://www.ic.unicamp.br/~helio/imagens_registro/

## Diagnóstico

Si encuentra problemas:

1. **"No se pueden cargar las imágenes"**: Verificar rutas y formatos (PNG para geométricas, JPEG para panorama)
2. **"No se encuentran suficientes puntos de interés"**: Las imágenes pueden ser muy similares o diferentes
3. **"Pocas correspondencias"**: Ajustar `-t/--ratio-threshold` (valores más altos son más permisivos)
4. **Panorama desalineado**: Ajustar `-r/--ransac-threshold` o usar método diferente `-f`

## Autores

Trabajo académico para el curso "Introducción al Procesamiento Digital de Imagen" (MC920/MO443)
Universidade Estadual de Campinas - Instituto de Computación
Profesor: Hélio Pedrini
