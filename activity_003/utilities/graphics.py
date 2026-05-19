import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from activity_003.utilities.load_save import save_images


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

    def view_multiple_analysis(
        self,
        results,
        title: str = "",
        plot: bool = True,
        save: bool = False,
        path_save: str = "",
        name_save: str = "",
    ):
        try:
            histograms = {}
            names = {
                "Original": "Original",
                "Translation": "Translação 50px em X e Y",
                "Rotation_45": "Rotação de 45 Graus",
                "Rotation_90": "Rotação de 90 Graus",
                "Scale": "Ampliação de 1,5x",
                "Reduction": "Redução de 0,5x",
            }

            fig = plt.figure(figsize=(8, 20))  # 20, 5
            gs = fig.add_gridspec(7, 2, height_ratios=[1, 1, 1, 1, 1, 1, 1])

            for idx, (name, item) in enumerate(results.items()):
                histograms[name] = (item["angles"], item["hist"])

                ax_rgb = fig.add_subplot(gs[idx, 0])
                ax_rgb.imshow(item["img_trans"], cmap="gray")
                ax_rgb.set_title(
                    f"Imagem: {names[name]}",
                    # fontsize=12,
                    fontdict={"fontstyle": "italic"},
                )
                ax_rgb.axis("off")

                ax_fft = fig.add_subplot(gs[idx, 1])
                ax_fft.imshow(item["mag_log"], cmap="inferno_r")
                ax_fft.set_title(
                    f"Espectro FFT: {names[name]}",
                    fontdict={"fontstyle": "italic"},
                )
                ax_fft.axis("off")

                h, w = item["mag_log"].shape
                cy, cx = h // 2, w // 2
                line_length = min(h, w) // 2
                arrow_length = line_length * 0.95

                x0 = cx - line_length * np.cos(item["dominant_angles"][0])
                y0 = cy + line_length * np.sin(item["dominant_angles"][0])
                x1 = cx + line_length * np.cos(item["dominant_angles"][0])
                y1 = cy - line_length * np.sin(item["dominant_angles"][0])

                ax_fft.plot(
                    [x0, x1],
                    [y0, y1],
                    color="#0FBAF7",
                    linewidth=1.5,
                    alpha=0.75,
                )

                ax_fft.arrow(
                    cx,
                    cy,
                    arrow_length * np.cos(item["dominant_angles"][0]),
                    -arrow_length * np.sin(item["dominant_angles"][0]),
                    head_width=h / 25,
                    head_length=h / 20,
                    fc="#0FBAF7",
                    ec="#0FBAF7",
                    linewidth=1.5,
                    alpha=0.75,
                    length_includes_head=True,
                )

                ax_fft.plot(cx, cy, "#0FBAF7", markersize=4, marker="o")

                if save:
                    save_images(
                        image=item["img_trans"].astype(np.uint8),
                        name_save=f"{name_save}_{name}_raw.png",
                        path=path_save,
                    )

                    min_val = np.min(item["mag_log"])
                    max_val = np.max(item["mag_log"])
                    mag_normalized = (item["mag_log"] - min_val) / (max_val - min_val)
                    mag_uint8 = (mag_normalized * 255).astype(np.uint8)

                    save_images(
                        image=mag_uint8,
                        name_save=f"{name_save}_{name}_fft.png",
                        path=path_save,
                    )

            ax_hist = fig.add_subplot(gs[6, :])
            for i, (name, (angles, hist)) in enumerate(histograms.items()):
                # Normalizar histograma para mejor comparación
                hist_norm = hist / np.max(hist)
                ax_hist.plot(angles, hist_norm, label=name, linewidth=1.5, alpha=0.75)

            ax_hist.set_title(
                "Comparação de histogramas de energia angular",
                fontsize=12,
                fontdict={"fontstyle": "italic"},
            )
            ax_hist.set_xlabel(r"Ângulo em radianos $[-\pi, \pi]$")
            ax_hist.set_ylabel("Energia Espectral Normalizada")
            ax_hist.grid(True, alpha=0.85)
            ax_hist.legend(loc="lower right", fontsize=8)

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
