import cv2
import numpy as np


class FeatureDetector:
    """Detector de puntos de interés y descriptores."""

    def __init__(self, method="sift"):
        """
        Inicializar detector de características.
        method: 'sift', 'surf', 'brief', 'orb'
        """
        self.method = method.lower()

        if self.method == "sift":
            self.detector = cv2.SIFT_create()
        elif self.method == "surf":
            self.detector = cv2.SURF_create()
        elif self.method == "orb":
            self.detector = cv2.ORB_create(nfeatures=5000)
        elif self.method == "brief":
            self.detector = cv2.ORB_create(nfeatures=5000)
        else:
            self.detector = cv2.SIFT_create()

    def detect_and_compute(self, img_gray):
        """Detectar puntos de interés y calcular descriptores."""
        keypoints, descriptors = self.detector.detectAndCompute(img_gray, None)
        return keypoints, descriptors


class FeatureMatcher:
    """Correspondencia de características entre dos imágenes."""

    @staticmethod
    def match_features(desc1, desc2, method="brute_force", ratio_threshold=0.7):
        """
        Encontrar correspondencias entre descriptores.
        method: 'brute_force' o 'flann'
        """
        if desc1 is None or desc2 is None or len(desc1) < 4 or len(desc2) < 4:
            return []

        if method == "flann":
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            matcher = cv2.FlannBasedMatcher(index_params, search_params)
            try:
                matches = matcher.knnMatch(desc1, desc2, k=2)
            except:
                return []
        else:
            matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            try:
                matches = matcher.knnMatch(desc1, desc2, k=2)
            except:
                return []

        if not matches:
            return []

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
            elif len(match_pair) == 1:
                good_matches.append(match_pair[0])

        return good_matches

    @staticmethod
    def extract_match_coordinates(kp1, kp2, matches):
        """Extraer coordenadas de puntos correspondientes."""
        if len(matches) < 4:
            return None, None

        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        return src_pts, dst_pts


class ImageRegistration:
    """Registro y alineación de imágenes."""

    @staticmethod
    def estimate_homography(src_pts, dst_pts, method="ransac", ransac_threshold=5.0):
        """Estimar matriz de homografía usando RANSAC."""
        if src_pts is None or dst_pts is None or len(src_pts) < 4:
            return None

        if method == "ransac":
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_threshold)
        else:
            H, mask = cv2.findHomography(src_pts, dst_pts, 0)

        return H, mask

    @staticmethod
    def warp_perspective(img, H, output_size):
        """Aplicar transformación de perspectiva a una imagen."""
        if H is None:
            return img

        warped = cv2.warpPerspective(img, H, output_size)
        return warped

    @staticmethod
    def _compute_transformed_bounds(h, w, H):
        """Calcular bounding box de una imagen después de transformación homográfica."""
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        corners_transformed = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), H)
        corners_transformed = corners_transformed.reshape(-1, 2)

        x_min = np.floor(corners_transformed[:, 0].min()).astype(int)
        y_min = np.floor(corners_transformed[:, 1].min()).astype(int)
        x_max = np.ceil(corners_transformed[:, 0].max()).astype(int)
        y_max = np.ceil(corners_transformed[:, 1].max()).astype(int)

        return x_min, y_min, x_max, y_max

    @staticmethod
    def _get_dense_bounds(h, w, H, grid_step=10):
        """Obtener bounding box más preciso usando malla de puntos."""
        y_coords = np.arange(0, h, grid_step, dtype=np.float32)
        x_coords = np.arange(0, w, grid_step, dtype=np.float32)

        grid_x, grid_y = np.meshgrid(x_coords, y_coords)
        grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

        grid_points_homogeneous = np.hstack(
            [grid_points, np.ones((len(grid_points), 1))]
        )
        transformed_points = (H @ grid_points_homogeneous.T).T

        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:3]

        x_min = np.floor(transformed_points[:, 0].min()).astype(int)
        y_min = np.floor(transformed_points[:, 1].min()).astype(int)
        x_max = np.ceil(transformed_points[:, 0].max()).astype(int)
        y_max = np.ceil(transformed_points[:, 1].max()).astype(int)

        return x_min, y_min, x_max, y_max

    @staticmethod
    def create_panorama(img1, img2, H):
        """Crear imagen panorámica preservando toda la información de ambas imágenes."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        # CORRECCIÓN: H mapea de img1 -> img2. Para traer img2 al espacio
        # de la imagen de referencia (img1), necesitamos su inversa.
        H_inv = np.linalg.inv(H)

        # CORRECCIÓN: Calculamos los límites de img2 en el espacio de img1 usando H_inv
        x_min_2, y_min_2, x_max_2, y_max_2 = ImageRegistration._get_dense_bounds(
            h2, w2, H_inv, grid_step=5
        )

        img1_bounds = np.array([0, 0, w1, h1])

        x_min = min(0, x_min_2)
        y_min = min(0, y_min_2)
        x_max = max(w1, x_max_2)
        y_max = max(h1, y_max_2)

        panorama_width = x_max - x_min
        panorama_height = y_max - y_min

        translation = np.array(
            [[1.0, 0.0, -x_min], [0.0, 1.0, -y_min], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        # CORRECCIÓN: La transformación total para img2 es aplicar H_inv (img2 -> img1)
        # y luego la traslación al nuevo lienzo expandido.
        H_translated = (translation @ H_inv).astype(np.float32)

        channels = img1.shape[2] if len(img1.shape) == 3 else 1
        dtype = img1.dtype

        panorama = np.zeros((panorama_height, panorama_width, channels), dtype=dtype)

        img1_translated = cv2.warpAffine(
            img1,
            translation[:2, :],
            (panorama_width, panorama_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        img2_warped = cv2.warpPerspective(
            img2,
            H_translated,
            (panorama_width, panorama_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        mask1 = cv2.warpAffine(
            np.ones((h1, w1), dtype=np.uint8) * 255,
            translation[:2, :],
            (panorama_width, panorama_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        mask2 = cv2.warpPerspective(
            np.ones((h2, w2), dtype=np.uint8) * 255,
            H_translated,
            (panorama_width, panorama_height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        mask1_bool = mask1 > 128
        mask2_bool = mask2 > 128

        overlap = mask1_bool & mask2_bool
        only_img1 = mask1_bool & ~mask2_bool
        only_img2 = mask2_bool & ~mask1_bool

        if len(panorama.shape) == 3:
            for c in range(channels):
                panorama[only_img1, c] = img1_translated[only_img1, c]
                panorama[only_img2, c] = img2_warped[only_img2, c]

                if overlap.sum() > 0:
                    w1_vals = img1_translated[overlap, c].astype(np.float32)
                    w2_vals = img2_warped[overlap, c].astype(np.float32)

                    mask1_overlap = mask1[overlap].astype(np.float32) / 255.0
                    mask2_overlap = mask2[overlap].astype(np.float32) / 255.0

                    alpha = mask2_overlap / (mask1_overlap + mask2_overlap + 1e-8)
                    blended = (1 - alpha) * w1_vals + alpha * w2_vals

                    panorama[overlap, c] = np.clip(blended, 0, 255).astype(dtype)
        else:
            panorama[only_img1] = img1_translated[only_img1]
            panorama[only_img2] = img2_warped[only_img2]

            if overlap.sum() > 0:
                w1_vals = img1_translated[overlap].astype(np.float32)
                w2_vals = img2_warped[overlap].astype(np.float32)

                mask1_overlap = mask1[overlap].astype(np.float32) / 255.0
                mask2_overlap = mask2[overlap].astype(np.float32) / 255.0

                alpha = mask2_overlap / (mask1_overlap + mask2_overlap + 1e-8)
                blended = (1 - alpha) * w1_vals + alpha * w2_vals

                panorama[overlap] = np.clip(blended, 0, 255).astype(dtype)

        return panorama, H_translated


class PanoramaBuilder:
    """Constructor de imágenes panorámicas."""

    def __init__(self, feature_method="sift", matcher_method="brute_force"):
        """Inicializar constructor de panorama."""
        self.feature_detector = FeatureDetector(feature_method)
        self.matcher_method = matcher_method
        self.matches = []
        self.H = None
        self.mask = None

    def register_images(
        self, img1_path, img2_path, ratio_threshold=0.7, ransac_threshold=5.0
    ):
        """Registrar un par de imágenes."""
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            raise ValueError("No se pueden cargar las imágenes")

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        kp1, desc1 = self.feature_detector.detect_and_compute(gray1)
        kp2, desc2 = self.feature_detector.detect_and_compute(gray2)

        if len(kp1) < 4 or len(kp2) < 4:
            raise ValueError("No se encontraron suficientes puntos de interés")

        self.matches = FeatureMatcher.match_features(
            desc1, desc2, self.matcher_method, ratio_threshold
        )

        if len(self.matches) < 4:
            raise ValueError(
                f"Solo se encontraron {len(self.matches)} correspondencias, se requieren al menos 4"
            )

        src_pts, dst_pts = FeatureMatcher.extract_match_coordinates(
            kp1, kp2, self.matches
        )

        self.H, self.mask = ImageRegistration.estimate_homography(
            src_pts, dst_pts, "ransac", ransac_threshold
        )

        if self.H is None:
            raise ValueError("No se pudo estimar la matriz de homografía")

        return {
            "img1": img1,
            "img2": img2,
            "gray1": gray1,
            "gray2": gray2,
            "kp1": kp1,
            "kp2": kp2,
            "matches": self.matches,
            "H": self.H,
        }

    def build_panorama(self, img1, img2):
        """Construir imagen panorámica a partir de dos imágenes registradas."""
        if self.H is None:
            raise ValueError("Primero debe ejecutar register_images")

        panorama, H_translated = ImageRegistration.create_panorama(img1, img2, self.H)

        return panorama

    def draw_matches(self, img1, img2, kp1, kp2, matches, output_path):
        """Dibujar líneas de correspondencia entre dos imágenes."""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]

        combined = np.zeros((max(h1, h2), w1 + w2, 3), dtype=img1.dtype)
        combined[:h1, :w1] = img1
        combined[:h2, w1:] = img2

        for match in matches:
            pt1 = tuple(map(int, kp1[match.queryIdx].pt))
            pt2 = (int(kp2[match.trainIdx].pt[0] + w1), int(kp2[match.trainIdx].pt[1]))

            cv2.line(combined, pt1, pt2, (0, 255, 0), 1)
            cv2.circle(combined, pt1, 4, (0, 0, 255), -1)
            cv2.circle(combined, pt2, 4, (255, 0, 0), -1)

        cv2.imwrite(output_path, combined)
        return combined


def process_panorama(
    img1_path,
    img2_path,
    matches_output_path,
    panorama_output_path,
    feature_method="sift",
    matcher_method="brute_force",
    ratio_threshold=0.7,
    ransac_threshold=5.0,
):
    """Procesar par de imágenes y crear panorama."""

    builder = PanoramaBuilder(feature_method, matcher_method)

    registration_data = builder.register_images(
        img1_path, img2_path, ratio_threshold, ransac_threshold
    )

    img1 = registration_data["img1"]
    img2 = registration_data["img2"]
    kp1 = registration_data["kp1"]
    kp2 = registration_data["kp2"]
    matches = registration_data["matches"]

    builder.draw_matches(img1, img2, kp1, kp2, matches, matches_output_path)

    panorama = builder.build_panorama(img1, img2)

    cv2.imwrite(panorama_output_path, panorama)

    return {
        "panorama": panorama,
        "matches": matches,
        "H": builder.H,
        "num_matches": len(matches),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Construcción de imagen panorámica")
    parser.add_argument("-i1", "--input1", required=True, help="Primera imagen (JPEG)")
    parser.add_argument("-i2", "--input2", required=True, help="Segunda imagen (JPEG)")
    parser.add_argument(
        "-m",
        "--matches",
        required=True,
        help="Salida de líneas de correspondencia (JPEG)",
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Salida de panorama (JPEG)"
    )
    parser.add_argument(
        "-f",
        "--feature",
        default="sift",
        choices=["sift", "surf", "orb", "brief"],
        help="Método de detección de características",
    )
    parser.add_argument(
        "-t",
        "--ratio-threshold",
        type=float,
        default=0.7,
        help="Umbral de ratio para filtrado de correspondencias",
    )
    parser.add_argument(
        "-r",
        "--ransac-threshold",
        type=float,
        default=5.0,
        help="Umbral de RANSAC para estimación de homografía",
    )

    args = parser.parse_args()

    result = process_panorama(
        args.input1,
        args.input2,
        args.matches,
        args.output,
        feature_method=args.feature,
        ratio_threshold=args.ratio_threshold,
        ransac_threshold=args.ransac_threshold,
    )

    print(f"Panorama creado exitosamente")
    print(f"Correspondencias encontradas: {result['num_matches']}")
    print(f"Homografía:\n{result['H']}")
