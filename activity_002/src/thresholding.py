# import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging

from activity_002.utilities.load_save import save_images
from activity_002.utilities.graphics import Graphics
import numpy as np

logger = logging.getLogger(__name__)
graphs = Graphics()


class Thresholding:
    def __init__(self, paths_res: dict, plot=False):
        self.paths_res = paths_res
        self.window_size = 15
        self.plot = plot
        pass

    # Global Thresholding
    def global_method(self, img, thresh: int, name_img: str = ""):
        ret, thresh_img = cv2.threshold(
            src=img, thresh=thresh, maxval=255, type=cv2.THRESH_BINARY
        )

        graphs.view_comparison_mult_hist(
            img_original=img,
            img_trasformada=thresh_img,
            title=f"Limiarização: Método Global $(T={ret})$",
            subtitle_imgt="Imagem com Limiarização",
            subtitle_histt="Imagem Limiarizada",
            save=True,
            path_save=self.paths_res["global_method"],
            name_save=f"thresholding_comparison_globalmethod_t{int(ret)}_{name_img}.png",
            plot=self.plot,
        )
        save_images(
            image=thresh_img,
            path=self.paths_res["global_method"],
            name_save=f"thresholding_globalmethod_t{int(ret)}_{name_img}.png",
        )

        del ret, thresh_img

    def otsu_method(self, img, name_img: str = ""):
        ret, thresh_img = cv2.threshold(
            img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        graphs.view_comparison_mult_hist(
            img_original=img,
            img_trasformada=thresh_img,
            title=f"Limiarização: Método Otsu $(T={ret})$",
            subtitle_imgt="Imagem com Limiarização",
            subtitle_histt="Imagem Limiarizada",
            save=True,
            path_save=self.paths_res["otsu_method"],
            name_save=f"thresholding_comparison_otsumethod_t{int(ret)}_{name_img}.png",
            plot=self.plot,
        )
        save_images(
            image=thresh_img,
            path=self.paths_res["otsu_method"],
            name_save=f"thresholding_otsumethod_t{int(ret)}_{name_img}.png",
        )

        del ret, thresh_img

    def average_method(self, img, thresh: int, name_img: str = ""):
        blockSize = self.window_size

        ret = (
            cv2.boxFilter(img, ddepth=-1, ksize=(blockSize, blockSize), normalize=True)
            - thresh
        )
        ret = int(np.mean(ret))

        thresh_img = cv2.adaptiveThreshold(
            src=img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,
            C=thresh,
        )

        graphs.view_comparison_mult_hist(
            img_original=img,
            img_trasformada=thresh_img,
            title=f"Limiarização: Método Média $(T={ret})$",
            subtitle_imgt="Imagem com Limiarização",
            subtitle_histt="Imagem Limiarizada",
            save=True,
            path_save=self.paths_res["average_method"],
            name_save=f"thresholding_comparison_averagemethod_t{int(ret)}_{name_img}.png",
            plot=self.plot,
        )
        save_images(
            image=thresh_img,
            path=self.paths_res["average_method"],
            name_save=f"thresholding_averagemethod_t{int(ret)}_{name_img}.png",
        )

        del ret, thresh_img

    def median_method(self, img, name_img: str = ""):
        median_val = np.median(img)
        ret, thresh_img = cv2.threshold(img, median_val, 255, cv2.THRESH_BINARY)

        graphs.view_comparison_mult_hist(
            img_original=img,
            img_trasformada=thresh_img,
            title=f"Limiarização: Método Mediana $(T={ret})$",
            subtitle_imgt="Imagem com Limiarização",
            subtitle_histt="Imagem Limiarizada",
            save=True,
            path_save=self.paths_res["median_method"],
            name_save=f"thresholding_comparison_medianmethod_t{int(ret)}_{name_img}.png",
            plot=self.plot,
        )
        save_images(
            image=thresh_img,
            path=self.paths_res["median_method"],
            name_save=f"thresholding_medianmethod_t{int(ret)}_{name_img}.png",
        )

        del ret, thresh_img

    # Métodos Locales/Adaptativos
    def niblack_method(self, img, thresh: int, name_img: str = ""):
        window_size = self.window_size

        mean = cv2.blur(img, (window_size, window_size))
        mean_sq = cv2.blur(img**2, (window_size, window_size))
        std = np.sqrt(np.abs(mean_sq - mean**2))
        ret = mean + thresh * std
        thresh_img = np.where(img > ret, 255, 0).astype(np.uint8)

        ret = int(np.mean(ret))

        graphs.view_comparison_mult_hist(
            img_original=img,
            img_trasformada=thresh_img,
            title=f"Limiarização: Método Niblack $(T={ret})$",
            subtitle_imgt="Imagem com Limiarização",
            subtitle_histt="Imagem Limiarizada",
            save=True,
            path_save=self.paths_res["niblack_method"],
            name_save=f"thresholding_comparison_niblackmethod_t{int(ret)}_{name_img}.png",
            plot=self.plot,
        )
        save_images(
            image=thresh_img,
            path=self.paths_res["niblack_method"],
            name_save=f"thresholding_niblackmethod_t{int(ret)}_{name_img}.png",
        )

        del ret, thresh_img, mean, mean_sq, std

    def sauvola_pietaksinen_method(self, img, thresh: int, name_img: str = ""):
        window_size = self.window_size
        R = 128

        mean = cv2.blur(img, (window_size, window_size))
        mean_sq = cv2.blur(img**2, (window_size, window_size))
        std = np.sqrt(np.abs(mean_sq - mean**2))

        ret = mean * (1 + thresh * (std / R - 1))
        thresh_img = np.where(img > ret, 255, 0).astype(np.uint8)

        ret = int(np.mean(ret))

        graphs.view_comparison_mult_hist(
            img_original=img,
            img_trasformada=thresh_img,
            title=f"Limiarização: Método Sauvola Pietaksinen $(T={ret})$",
            subtitle_imgt="Imagem com Limiarização",
            subtitle_histt="Imagem Limiarizada",
            save=True,
            path_save=self.paths_res["sauvola_pietaksinen_method"],
            name_save=f"thresholding_comparison_sauvolapietaksinenmethod_t{int(ret)}_{name_img}.png",
            plot=self.plot,
        )
        save_images(
            image=thresh_img,
            path=self.paths_res["sauvola_pietaksinen_method"],
            name_save=f"thresholding_sauvola_pietaksinenmethod_t{int(ret)}_{name_img}.png",
        )

        del ret, thresh_img, mean, mean_sq, std

    def contrast_method(self, img, thresh: int, name_img: str = ""):
        window_size = self.window_size

        kernel = np.ones((window_size, window_size), np.uint8)
        local_max = cv2.dilate(img, kernel)
        local_min = cv2.erode(img, kernel)

        thresh_img = np.where((local_max - local_min) > thresh, 255, 0).astype(np.uint8)

        graphs.view_comparison_mult_hist(
            img_original=img,
            img_trasformada=thresh_img,
            title=f"Limiarização: contrast $(T={thresh})$",
            subtitle_imgt="Imagem com Limiarização",
            subtitle_histt="Imagem Limiarizada",
            save=True,
            path_save=self.paths_res["contrast_method"],
            name_save=f"thresholding_comparison_contrastmethod_t{int(thresh)}_{name_img}.png",
            plot=self.plot,
        )
        save_images(
            image=thresh_img,
            path=self.paths_res["contrast_method"],
            name_save=f"thresholding_sauvola_contrastmethod_t{int(thresh)}_{name_img}.png",
        )

        del thresh_img, kernel, local_max, local_min

    def bernsen_method(self, img, name_img: str = ""):

        window_size = self.window_size

        kernel = np.ones((window_size, window_size), np.uint8)
        local_max = cv2.dilate(img, kernel)
        local_min = cv2.erode(img, kernel)

        ret = (local_max.astype(float) + local_min.astype(float)) / 2

        thresh_img = np.where(img > ret, 255, 0).astype(np.uint8)
        ret = int(np.mean(ret))

        graphs.view_comparison_mult_hist(
            img_original=img,
            img_trasformada=thresh_img,
            title=f"Limiarização: Método Bernsen $(T={ret})$",
            subtitle_imgt="Imagem com Limiarização",
            subtitle_histt="Imagem Limiarizada",
            save=True,
            path_save=self.paths_res["bernsen_method"],
            name_save=f"thresholding_comparison_bernsenmethod_t{int(ret)}_{name_img}.png",
            plot=self.plot,
        )
        save_images(
            image=thresh_img,
            path=self.paths_res["bernsen_method"],
            name_save=f"thresholding_sauvola_bernsenmethod_t{int(ret)}_{name_img}.png",
        )

        del ret, thresh_img, kernel, local_max, local_min

    def phansalskar_more_sabale_method(self, img, thresh: int, name_img: str = ""):
        p, q = 2.0, 10.0
        R = 128
        window_size = self.window_size

        mean = cv2.blur(img, (window_size, window_size))
        mean_sq = cv2.blur(img**2, (window_size, window_size))
        std = np.sqrt(np.abs(mean_sq - mean**2))

        ret = mean * (1 + p * np.exp(-q * mean) + thresh * (std / R - 1))
        thresh_img = np.where(img > ret, 255, 0).astype(np.uint8)

        ret = int(np.mean(ret))
        graphs.view_comparison_mult_hist(
            img_original=img,
            img_trasformada=thresh_img,
            title=f"Limiarização: Método Phansalskar More Sabale $(T={ret})$",
            subtitle_imgt="Imagem com Limiarização",
            subtitle_histt="Imagem Limiarizada",
            save=True,
            path_save=self.paths_res["phansalskar_more_sabale_method"],
            name_save=f"thresholding_comparison_phansalskarmoresabalemethod_t{int(ret)}_{name_img}.png",
            plot=self.plot,
        )
        save_images(
            image=thresh_img,
            path=self.paths_res["phansalskar_more_sabale_method"],
            name_save=f"thresholding_sauvola_phansalskarmoresabalemethod_t{int(ret)}_{name_img}.png",
        )

        del ret, thresh_img, mean, mean_sq, std
