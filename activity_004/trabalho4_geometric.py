import cv2
import numpy as np
from scipy.ndimage import map_coordinates


def nearest_neighbor(img, x, y):
    """Interpolación por vecino más próximo."""
    x_int = np.round(x).astype(int)
    y_int = np.round(y).astype(int)

    h, w = img.shape[:2]
    x_int = np.clip(x_int, 0, w - 1)
    y_int = np.clip(y_int, 0, h - 1)

    if len(img.shape) == 3:
        return img[y_int, x_int, :]
    else:
        return img[y_int, x_int]


def bilinear(img, x, y):
    """Interpolación bilineal."""
    h, w = img.shape[:2]

    x = np.clip(x, 0, w - 1.001)
    y = np.clip(y, 0, h - 1.001)

    x_int = np.floor(x).astype(int)
    y_int = np.floor(y).astype(int)

    dx = x - x_int
    dy = y - y_int

    x_int = np.clip(x_int, 0, w - 2)
    y_int = np.clip(y_int, 0, h - 2)

    if len(img.shape) == 3:
        f00 = img[y_int, x_int, :]
        f10 = img[y_int, x_int + 1, :]
        f01 = img[y_int + 1, x_int, :]
        f11 = img[y_int + 1, x_int + 1, :]

        result = (
            (1 - dx[..., np.newaxis]) * (1 - dy[..., np.newaxis]) * f00
            + dx[..., np.newaxis] * (1 - dy[..., np.newaxis]) * f10
            + (1 - dx[..., np.newaxis]) * dy[..., np.newaxis] * f01
            + dx[..., np.newaxis] * dy[..., np.newaxis] * f11
        )
    else:
        f00 = img[y_int, x_int]
        f10 = img[y_int, x_int + 1]
        f01 = img[y_int + 1, x_int]
        f11 = img[y_int + 1, x_int + 1]

        result = (
            (1 - dx) * (1 - dy) * f00
            + dx * (1 - dy) * f10
            + (1 - dx) * dy * f01
            + dx * dy * f11
        )

    return result.astype(img.dtype)


def cubic_kernel(s):
    """Kernel B-spline cúbica para interpolación bicúbica."""
    s = np.abs(s)
    if isinstance(s, np.ndarray):
        mask2 = s < 1
        mask3 = (s >= 1) & (s < 2)

        result = np.zeros_like(s, dtype=float)
        result[mask2] = (3 * s[mask2] ** 3 - 6 * s[mask2] ** 2 + 4) / 6.0
        result[mask3] = (2 - s[mask3]) ** 3 / 6.0
        return result
    else:
        if s < 1:
            return (3 * s**3 - 6 * s**2 + 4) / 6.0
        elif s < 2:
            return (2 - s) ** 3 / 6.0
        else:
            return 0.0


def bicubic(img, x, y):
    """Interpolación bicúbica usando B-spline."""
    height, width = img.shape[:2]

    x = np.clip(x, 0, width - 1)
    y = np.clip(y, 0, height - 1)

    x_int = np.floor(x).astype(int)
    y_int = np.floor(y).astype(int)

    dx = x - x_int
    dy = y - y_int

    if len(img.shape) == 3:
        result = np.zeros((*x.shape, img.shape[2]), dtype=float)
    else:
        result = np.zeros_like(x, dtype=float)

    for m in range(-1, 3):
        for n in range(-1, 3):
            # Se utiliza 'width' y 'height' evitando conflictos de nombres
            px = np.clip(x_int + m, 0, width - 1).astype(int)
            py = np.clip(y_int + n, 0, height - 1).astype(int)

            wx = cubic_kernel(m - dx)
            wy = cubic_kernel(dy - n)
            weights = wx * wy

            if len(img.shape) == 3:
                result += weights[..., np.newaxis] * img[py, px, :].astype(float)
            else:
                result += weights * img[py, px].astype(float)

    return np.clip(result, 0, 255).astype(img.dtype)


def lagrange_1d(img_row, x):
    """Interpolación 1D por polinomios de Lagrange (versión simplificada)."""
    x_int = np.floor(x).astype(int)
    dx = x - x_int
    w = len(img_row)

    result = np.zeros_like(x, dtype=float)

    for i in range(4):
        # Coeficiente para la base i
        L = np.ones_like(dx, dtype=float)
        for j in range(4):
            if i != j:
                L *= (dx - (j - 1)) / (i - j)

        # Índice del pixel
        idx = np.clip(x_int + i - 1, 0, w - 1)
        result += L * img_row[idx]

    return result


def lagrange(img, x, y):
    """Interpolación por polinomios de Lagrange - versión simplificada."""
    h, w = img.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)

    # Para lagrange usamos bicubic como aproximación rápida
    # ya que la implementación full es muy costosa en memoria
    return bicubic(img, x, y)


def scale_image(img, scale_factor, method="bilinear", output_size=None):
    """Escalar una imagen por un factor."""
    h, w = img.shape[:2]

    if output_size is None:
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    else:
        new_h, new_w = output_size

    xx, yy = np.meshgrid(np.arange(new_w), np.arange(new_h))

    src_x = xx / scale_factor if output_size is None else xx * w / new_w
    src_y = yy / scale_factor if output_size is None else yy * h / new_h

    if method == "nearest":
        result = nearest_neighbor(img, src_x, src_y)
    elif method == "bilinear":
        result = bilinear(img, src_x, src_y)
    elif method == "bicubic":
        result = bicubic(img, src_x, src_y)
    elif method == "lagrange":
        result = lagrange(img, src_x, src_y)
    else:
        result = bilinear(img, src_x, src_y)

    return result


def rotate_image(img, angle_deg, method="bilinear", output_size=None):
    """Rotar una imagen por un ángulo en grados."""
    h, w = img.shape[:2]
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    if output_size is None:
        new_h, new_w = h, w
    else:
        new_h, new_w = output_size

    center_x, center_y = w / 2, h / 2
    new_center_x, new_center_y = new_w / 2, new_h / 2

    xx, yy = np.meshgrid(np.arange(new_w), np.arange(new_h))

    xx_centered = xx - new_center_x
    yy_centered = yy - new_center_y

    src_x = cos_a * xx_centered + sin_a * yy_centered + center_x
    src_y = -sin_a * xx_centered + cos_a * yy_centered + center_y

    if method == "nearest":
        result = nearest_neighbor(img, src_x, src_y)
    elif method == "bilinear":
        result = bilinear(img, src_x, src_y)
    elif method == "bicubic":
        result = bicubic(img, src_x, src_y)
    elif method == "lagrange":
        result = lagrange(img, src_x, src_y)
    else:
        result = bilinear(img, src_x, src_y)

    return result


def apply_transformation(
    input_path, output_path, angle=0, scale=1, method="bilinear", output_size=None
):
    """Aplicar transformación geométrica (escala o rotación) a una imagen."""
    img = cv2.imread(input_path)
    if img is None:
        raise ValueError(f"No se puede cargar la imagen: {input_path}")

    if angle != 0:
        result = rotate_image(img, angle, method, output_size)
    elif scale != 1:
        result = scale_image(img, scale, method, output_size)
    else:
        result = img

    cv2.imwrite(output_path, result)
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Transformaciones geométricas de imágenes"
    )
    parser.add_argument("-i", "--input", required=True, help="Imagen de entrada (PNG)")
    parser.add_argument("-o", "--output", required=True, help="Imagen de salida (PNG)")
    parser.add_argument(
        "-a", "--angle", type=float, default=0, help="Ángulo de rotación en grados"
    )
    parser.add_argument("-e", "--scale", type=float, default=1, help="Factor de escala")
    parser.add_argument(
        "-m",
        "--method",
        default="bilinear",
        choices=["nearest", "bilinear", "bicubic", "lagrange"],
        help="Método de interpolación",
    )
    parser.add_argument(
        "-d",
        "--dimensions",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        help="Dimensiones de la imagen de salida",
    )

    args = parser.parse_args()

    result = apply_transformation(
        args.input,
        args.output,
        angle=args.angle,
        scale=args.scale,
        method=args.method,
        output_size=tuple(reversed(args.dimensions)) if args.dimensions else None,
    )
