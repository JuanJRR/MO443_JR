import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


class Graphics:
    """
    A helper class for managing global matplotlib styling and providing simplified methods for image visualization.
    """

    def __init__(self):
        """
        Initializes the visualization environment with a professional aesthetic inspired by scientific publications.

        *   Styling: Utilizes ggplot as a base style.
        *   Typography: Configures "Latin Modern Roman" (serif) to ensure high-quality text rendering, particularly useful for reports.
        *   Theming: Sets a clean white background with subtle light-gray grid lines (#E5E5E5).
        *   Image Defaults: Disables interpolation by default to show raw pixel values accurately.
        """

        plt.style.use("ggplot")
        plt.rcParams.update(
            {
                # Background and Grid
                "axes.facecolor": "white",
                "axes.edgecolor": "white",
                "grid.color": "#E5E5E5",
                "grid.linestyle": "-",
                # Typography and Labels
                "font.family": "serif",
                "font.sans-serif": ["Latin Modern Roman", "LM Roman 10"],
                "axes.titlesize": 10,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                # Default image (Perceptual Cmap)
                "image.interpolation": "none",
            }
        )

    def view_comparison_mult_hist(
        self,
        img_original,
        img_trasformada,
        title: str = "",
        subtitle_imgt: str = "Imagen con Umbralización",
        subtitle_histt: str = "Imagem Limiarizada",
        plot: bool = True,
        save: bool = False,
        path_save: str = "",
        name_save: str = "original_image",
    ):
        try:
            fig, axs = plt.subplots(1, 4, figsize=(20, 5))
            fig.suptitle(title, fontsize=12, fontdict={"fontstyle": "italic"})

            # 1. Histograma Imagen Original (Izquierda)
            axs[0].hist(img_original.ravel(), bins=256, range=[0, 256], color="gray")
            axs[0].set_title("Histograma: Imagem Original")
            axs[0].set_xlim([0, 256])
            axs[0].set_xlabel("Densidade")
            axs[0].set_ylabel("Níveis de Cinza")

            # 2. Imagen Original
            axs[1].imshow(img_original, cmap="gray")
            axs[1].set_title("Imagen Original")
            axs[1].axis("off")

            # 3. Imagen Traformada
            axs[2].imshow(img_trasformada, cmap="gray")
            axs[2].set_title(subtitle_imgt)
            axs[2].axis("off")

            # 4. Histograma Imagen Binarizada (Derecha)
            # Tras binarizar, los valores se concentran en 0 y 255
            axs[3].hist(
                img_trasformada.ravel(), bins=256, range=[0, 256], color="black"
            )
            axs[3].set_title(f"Histograma: {subtitle_histt}")
            axs[3].set_xlim([0, 256])
            axs[0].set_xlabel("Densidade")
            axs[0].set_ylabel("Níveis de Cinza")

            plt.tight_layout()

            if save:
                path_save_im = Path(os.path.join(path_save, name_save)).resolve()
                plt.savefig(
                    path_save_im,
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.05,
                    transparent=False,
                    facecolor="white",
                )

            if plot:
                plt.show()

        finally:
            plt.close(fig)
            pass
        pass

    def view_transition_hist(
        self,
        diferencias,
        indices_transicion,
        valores_transicion,
        T,
        alpha,
        title: str = "",
        plot: bool = True,
        save: bool = False,
        path_save: str = "",
        name_save: str = "original_image",
    ):
        try:
            fig, axs = plt.subplots(figsize=(8, 5))
            axs.plot(
                diferencias,
                label="Diferença entre histogramas",
                alpha=0.85,
                color = "gray"
            )
            axs.axhline(y=T, color="red", linestyle="--", label=f"Limiar $T (α={alpha})$", alpha=0.65)

            # Agregar marcadores (puntos) donde se produjo la transición
            axs.scatter(
                np.array(indices_transicion) - 1,
                valores_transicion,
                color="green",
                marker="x",
                s=100,
                label="Transições detectadas",
                alpha=0.65
            )

            axs.set_title(title, fontdict={"fontstyle": "italic"})
            axs.set_xlabel("Quadro")
            axs.set_ylabel("Medida $(D_i)$")
            axs.legend()
            axs.grid(True, linestyle=":", alpha=0.7)
            axs.axis("on")

            plt.tight_layout()

            if save:
                path_save_im = Path(os.path.join(path_save, name_save)).resolve()
                plt.savefig(
                    path_save_im,
                    dpi=600,
                    bbox_inches="tight",
                    pad_inches=0.05,
                    transparent=False,
                    facecolor="white",
                )

            if plot:
                plt.show()

        finally:
            plt.close(fig)
            pass
        pass
