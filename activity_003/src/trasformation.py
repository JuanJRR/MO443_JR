import cv2
import numpy as np
from scipy.signal import find_peaks

from activity_003.utilities.load_save import upload_images


class TrasformationFFT2D:
    def __init__(
        self, path_img: str, name_save: str = "", save: bool = False, plot: bool = False
    ):

        self.img = upload_images(path=path_img, color=False)
        self.num_bins = 36
        self.center_radius = 20
        self.num_peaks = 2

    def __compute_fft2d(self, img):
        # Cálculo de la FFT y centrado del espectro
        f_transform = np.fft.fft2(img)
        f_shift = np.fft.fftshift(f_transform)

        # Magnitud (escala logarítmica para visualización) y Fase
        magnitude = np.abs(f_shift)
        magnitude_log = np.log(1 + magnitude)
        phase = np.angle(f_shift)

        return magnitude_log, phase, f_shift

    def __compute_angular_histogram(self, magnitude):

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2

        # Crear mallas de coordenadas vectorizadas
        y, x = np.indices((h, w))

        # Coordenadas relativas al centro (y invertido para alineación matemática estándar)
        dy = cy - y
        dx = x - cx

        # Calcular ángulos y radios vectorialmente
        angles = np.arctan2(dy, dx)  # Rango [-pi, pi]
        radii = np.sqrt(dx**2 + dy**2)

        # Máscara para ignorar bajas frecuencias (centro)
        mask = radii > self.center_radius

        valid_angles = angles[mask]
        valid_energy = magnitude[mask]

        # Histograma de energía angular (acumulación de magnitudes)
        hist, bin_edges = np.histogram(
            valid_angles,
            bins=self.num_bins,
            range=(-np.pi, np.pi),
            weights=valid_energy,
        )

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, hist

    def __find_dominant_orientations(self, bin_centers, hist):
        # Suavizar un poco el histograma para evitar picos ruidosos
        kernel = np.ones(3) / 3
        hist_smooth = np.convolve(hist, kernel, mode="same")

        peaks, _ = find_peaks(hist_smooth, distance=len(hist) // 8)

        # Ordenar picos por energía descendente
        peak_energies = hist_smooth[peaks]
        sorted_peak_indices = peaks[np.argsort(peak_energies)[::-1]]

        dominant_angles = bin_centers[sorted_peak_indices[: self.num_peaks]]
        return dominant_angles

    def __apply_transformations(self):
        h, w = self.img.shape
        center = (w // 2, h // 2)

        # 1. Original
        transforms = {"Original": self.img}

        # 2. Traslación (Shift de 50px en X y Y)
        M_trans = np.float32([[1, 0, 50], [0, 1, 50]])
        gray_trans = cv2.warpAffine(self.img, M_trans, (w, h))
        transforms["Translation"] = gray_trans

        # 3. Rotación (45 grados)
        M_rot = cv2.getRotationMatrix2D(center, 45, 1.0)
        gray_rot_45 = cv2.warpAffine(self.img, M_rot, (w, h))
        transforms["Rotation_45"] = gray_rot_45

        # 4. Rotación (90 grados)
        M_rot = cv2.getRotationMatrix2D(center, 90, 1.0)
        gray_rot_90 = cv2.warpAffine(self.img, M_rot, (w, h))
        transforms["Rotation_90"] = gray_rot_90

        # 4. Escala
        M_scale = cv2.getRotationMatrix2D(center, 0, 1.5)
        gray_scale = cv2.warpAffine(self.img, M_scale, (w, h))
        transforms["Scale"] = gray_scale

        # 4. Reduccion
        M_scale = cv2.getRotationMatrix2D(center, 0, 0.5)
        gray_reduction = cv2.warpAffine(self.img, M_scale, (w, h))
        transforms["Reduction"] = gray_reduction

        return transforms

    def comparative_analysis(self):
        results = {}

        transformation = [
            "Original",
            "Translation",
            "Rotation_45",
            "Rotation_90",
            "Scale",
            "Reduction",
        ]

        transforms = self.__apply_transformations()

        for item in transformation:
            img_t = transforms[item]
            mag_log, _, _ = self.__compute_fft2d(img=img_t)

            angles, hist = self.__compute_angular_histogram(magnitude=mag_log)

            dominant_angles = self.__find_dominant_orientations(
                bin_centers=angles,
                hist=hist,
            )

            results[item] = {
                "img_trans": img_t,
                "mag_log": mag_log,
                "hist": hist,
                "angles": angles,
                "dominant_angles": dominant_angles,
            }

        return results
