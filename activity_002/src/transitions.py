import cv2
import numpy as np
import os

import logging
from pathlib import Path

import matplotlib.pyplot as plt
from activity_002.utilities.graphics import Graphics

logger = logging.getLogger(__name__)
graph = Graphics()


class Transitions:
    def __init__(self, paths_res: dict, plot=False):
        self.paths_res = paths_res
        self.plot = plot

    def detect_pixels(self, video_path: str, T1=15, T2_factor=0.01, name_img: str = ""):

        logging.info("Pixel difference segmentation process initiated.")

        cap = cv2.VideoCapture(Path(video_path).resolve())
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_pixeles = width * height

        T2 = T2_factor * total_pixeles

        ret, prev_frame = cap.read()
        if not ret:
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        indices_transicion = []
        frame_count = 1

        logger.info(f"Analyzing video: {video_path}...")
        logger.info("Looking for transitions where {T1} > {T2}.")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            diff = cv2.absdiff(prev_gray, curr_gray)
            píxeles_alterados = np.sum(diff > T1)

            if píxeles_alterados > T2:
                indices_transicion.append(frame_count)
                logger.info(f"Transition detected in the frame: {frame_count}")

            prev_gray = curr_gray
            frame_count += 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        margen = 4
        for i, idx in enumerate(indices_transicion):
            inicio = max(0, idx - int(margen))
            fin = min(frame_count, idx + int(margen) + 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)

            nombre_archivo = Path(
                os.path.join(
                    self.paths_res["differences_between_pixels"],
                    f"dpixel_{i + 1:03d}_frame_{idx}_{name_img}.mp4",
                )
            ).resolve()
            out = cv2.VideoWriter(nombre_archivo, fourcc, fps, (width, height))

            for _ in range(inicio, fin):
                ret_extra, frame_extra = cap.read()
                if not ret_extra:
                    break
                out.write(frame_extra)

            out.release()
            logging.info(f"Stored transitions: {nombre_archivo}")

        cap.release()
        logging.info("Pixel difference segmentation process completed.")

    def detect_blocks(
        self, video_path: str, block_size=16, T1=15, T2_factor=0.01, name_img: str = ""
    ):
        logging.info("Block difference segmentation process initiated.")

        cap = cv2.VideoCapture(Path(video_path).resolve())
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_pixeles = width * height

        T2 = T2_factor * total_pixeles

        ret, prev_frame = cap.read()
        if not ret:
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        indices_transicion = []
        frame_count = 1

        logger.info(f"Analyzing video: {video_path}...")
        logger.info(
            "Looking for transitions where block ({block_size}x{block_size}). {T1} > {T2}."
        )

        # Fase 1: Identificación de puntos de corte
        bloques_h = height // block_size
        bloques_w = width // block_size
        total_bloques = bloques_h * bloques_w
        umbral_T2 = T2_factor * total_bloques

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            bloques_distintos = 0

            # Iteración sobre los bloques de la imagen
            for y in range(0, bloques_h * block_size, block_size):
                for x in range(0, bloques_w * block_size, block_size):
                    # Extraer bloques correspondientes
                    b_prev = prev_gray[y : y + block_size, x : x + block_size]
                    b_curr = curr_gray[y : y + block_size, x : x + block_size]

                    # Calcular Error Cuadrático Medio (MSE)
                    mse = np.mean((b_prev.astype(float) - b_curr.astype(float)) ** 2)

                    if mse > T1:
                        bloques_distintos += 1

            # Si el número de bloques distintos excede T2
            if bloques_distintos > umbral_T2:
                indices_transicion.append(frame_count)
                print(f"Transition detected in the frame:: {frame_count}")

            prev_gray = curr_gray
            frame_count += 1

        # Fase 2: Creación de Micro-Videos (.mp4)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        margen = 4
        for i, idx in enumerate(indices_transicion):
            # Definimos el rango del micro-video (1 seg antes y 1 seg después)
            inicio = max(0, idx - int(margen))
            fin = min(frame_count, idx + int(margen) + 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)

            nombre_archivo = Path(
                os.path.join(
                    self.paths_res["differences_between_blocks"],
                    f"dblock_{i + 1:03d}_frame_{idx}_b{block_size}_{name_img}.mp4",
                )
            ).resolve()
            out = cv2.VideoWriter(nombre_archivo, fourcc, fps, (width, height))

            for _ in range(inicio, fin):
                ret_extra, frame_extra = cap.read()
                if not ret_extra:
                    break
                out.write(frame_extra)

            out.release()
            logging.info(f"Stored transitions: {nombre_archivo}")

        cap.release()
        logging.info("Block difference segmentation process completed.")

    def detect_histograms(self, video_path: str, alpha=4, bins=256, name_img: str = ""):
        logging.info("Histogram difference segmentation process initiated.")

        cap = cv2.VideoCapture(Path(video_path).resolve())
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        ret, prev_frame = cap.read()
        if not ret:
            return

        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        diferencias = []

        # --- FASE 1: Cálculo de la métrica (D_i) ---
        print("Calculando diferencias de histogramas...")
        for i in range(1, total_f):  # noqa: B007
            ret, frame = cap.read()
            if not ret:
                break

            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Histograma de 256 niveles (j=1 hasta B) [cite: 86, 87]
            h1 = cv2.calcHist([prev_gray], [0], None, [bins], [0, bins])
            h2 = cv2.calcHist([curr_gray], [0], None, [bins], [0, bins])

            # Sumatoria de diferencias absolutas (D_i)
            diff = np.sum(np.abs(h1 - h2))
            diferencias.append(diff)

            prev_gray = curr_gray

        # --- FASE 2: Detección con Umbral Dinámico ---
        mu = np.mean(diferencias)
        sigma = np.std(diferencias)
        T = mu + alpha * sigma  # Umbral T = mu + alpha*sigma [cite: 90]

        indices_transicion = [i + 1 for i, d in enumerate(diferencias) if d > T]
        valores_transicion = [diferencias[i - 1] for i in indices_transicion]

        # Fase 2: Creación de Micro-Videos (.mp4)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        margen = 4
        for i, idx in enumerate(indices_transicion):
            # Definimos el rango del micro-video (1 seg antes y 1 seg después)
            inicio = max(0, idx - int(margen))
            fin = min(total_f, idx + int(margen) + 1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)

            nombre_archivo = Path(
                os.path.join(
                    self.paths_res["differences_between_histograms"],
                    f"dhist_{i + 1:03d}_frame_{idx}_{name_img}.mp4",
                )
            ).resolve()
            out = cv2.VideoWriter(nombre_archivo, fourcc, fps, (width, height))

            for _ in range(inicio, fin):
                ret_extra, frame_extra = cap.read()
                if not ret_extra:
                    break
                out.write(frame_extra)

            out.release()
            logging.info(f"Stored transitions: {nombre_archivo}")

        cap.release()

        graph.view_transition_hist(
            diferencias=diferencias,
            indices_transicion=indices_transicion,
            valores_transicion=valores_transicion,
            alpha=alpha,
            T=T,
            title=f"Histograma de detección de transiciones ({name_img})",
            save=True,
            path_save=self.paths_res["differences_between_histograms"],
            name_save=f"{name_img}_hist.png",
            plot=False
        )

        logging.info("Histogram difference segmentation process completed.")
