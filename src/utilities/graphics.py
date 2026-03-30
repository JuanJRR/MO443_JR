import os

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
                "axes.titlesize": 11,
                "axes.labelsize": 9,
                "xtick.labelsize": 8,
                "ytick.labelsize": 8,
                # Default image (Perceptual Cmap)
                "image.interpolation": "none",
            }
        )

    def view_simplified(
        self,
        image: np.ndarray,
        title: str = "Original image",
        save: bool = False,
        path_save: str = "/app/experiments/activity_001/results",
        name_save: str = "Original_image",
        plot: bool = True,
        bar: bool = False,
    ):
        """
        A robust method to display or save a single image with pre-configured axes, labels, and colorbars.

        Args:
            image (numpy.ndarray): The input image (NumPy array) to be visualized.
            title (str, optional): Title displayed at the top of the plot in italics. Defaults to "Original image".
            save (bool, optional): Whether to export the figure to a file. Defaults to False.
            path_save (str, optional): Directory where the image will be saved. Defaults to "codes/activity_001/results".
            name_save (str, optional): Filename for the saved figure. Defaults to "Original image".
            plot (bool, optional): Whether to trigger plt.show() to display the image in the UI. Defaults to True.

        Raises:
            Exception: If an error occurs during image plotting or saving.

        Note:
            The method ensures the figure is closed after execution to prevent
            memory leaks in batch processing.
        """

        fig, ax = plt.subplots(figsize=(5, 5))

        try:
            im = ax.imshow(
                image, aspect="equal", cmap="gray" if image.ndim == 2 else "viridis"
            )

            ax.set_title(title, pad=10, fontdict={"fontstyle": "italic"})
            ax.set_xlabel("Largo (px)")
            ax.set_ylabel("Longo (px)")

            ax.grid(
                True,
                color="white",
                linestyle="-",
                linewidth=0.5,
                alpha=0.3,
            )

            if bar:
                cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=9)
                cbar.outline.set_visible(False)

            plt.tight_layout()

            if save:
                path_save_im = os.path.join(path_save, name_save)
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

    def view_comparison(
        self,
        img_orig,
        img_trand,
        factor: float = 1,
        info_trasd: str = "Transformação",
        save: bool = False,
        path_save: str = "/app/experiments/activity_001/results",
        name_save: str = "Original_image",
        plot: bool = True,
        bar: bool = False,
    ):
        """Displays two images side-by-side for comparative analysis.

        This method creates a multi-panel figure where the original image and
        its transformed version are shown together. It automatically adjusts
        the colormap based on the image dimensions (Grayscale for 2D arrays
        or Viridis for 3D/color arrays).s

            Args:
                img_orig (numpy.ndarray): The base or original image array.
                img_trand (numpy.ndarray): The transformed or processed image array.
                factor (float, optional): Aspect ratio adjustment factor for the second plot's width. Defaults to 1.
                info_trasd (str, optional): Descriptive title for the second image
                (e.g., 'Median Filter', 'Sobel Edge'). Defaults to "Transformação".
                save (bool, optional): Whether to export the comparison figure to a file. Defaults to False.
                path_save (str, optional): Destination directory for the saved file. Defaults to "/app/experiments/activity_001/results".
                name_save (str, optional): Filename for the output figure. Defaults to "Original_image".
                plot (bool, optional): Whether to trigger plt.show() to display
                the UI. Defaults to True.
                bar (bool, optional): If True, adds a unified colorbar to the
                transformed image. Defaults to False.

            Note:
            - When saving, it uses a high resolution of 600 DPI, suitable for
              technical reports.
        """
        try:
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(8, 6), gridspec_kw={"width_ratios": [1, factor]}
            )

            im1 = ax1.imshow(  # noqa: F841
                img_orig,
                cmap="gray" if img_orig.ndim == 2 else "viridis",
            )
            ax1.set_title(
                "Imagem original",
                fontsize=10,
                fontdict={"fontstyle": "italic"},
            )
            ax1.tick_params(labelsize=9)

            im2 = ax2.imshow(
                img_trand,
                cmap="gray" if img_trand.ndim == 2 else "viridis",
            )
            ax2.set_title(
                f"{info_trasd}",
                fontsize=10,
                fontdict={"fontstyle": "italic"},
            )
            ax2.tick_params(labelsize=9)

            ax1.set_xlabel("Largo (px)")
            ax2.set_xlabel("Largo (px)")

            ax1.set_ylabel("Longo (px)")
            ax2.set_ylabel("Longo (px)")

            ax1.grid(
                True,
                color="white",
                linestyle="-",
                linewidth=0.5,
                alpha=0.3,
            )

            ax2.grid(
                True,
                color="white",
                linestyle="-",
                linewidth=0.5,
                alpha=0.3,
            )

            if bar:
                divider = make_axes_locatable(ax2)
                cax = divider.append_axes("right", size="3%", pad=0.15)
                cbar = fig.colorbar(
                    im2,
                    ax=[ax1, ax2],
                    orientation="vertical",
                    fraction=0.02,
                    pad=0.04,
                    cax=cax,
                )
                cbar.ax.tick_params(labelsize=9)
                cbar.outline.set_visible(False)

            plt.tight_layout()

            if save:
                path_save_im = os.path.join(path_save, name_save)
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
