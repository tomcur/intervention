from pathlib import Path

import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np

DPI = 300
matplotlib.use("Agg")
matplotlib.rcParams["figure.dpi"] = DPI


def print_multipage(path: Path, tick: int, rgb_image: np.ndarray, heatmaps: np.ndarray):
    """
    Prints a multi-page PDF.
    """
    with PdfPages(path) as pdf:
        height, width, _channels = rgb_image.shape
        figsize = width / float(DPI), height / float(DPI)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis("off")
        ax.imshow(rgb_image)
        pdf.attach_note("RGB camera image")
        pdf.savefig(fig)
        plt.close()

        rescaled_heatmaps = np.power(1.16, np.log(heatmaps))
        heatmaps_shape = rescaled_heatmaps.shape

        for command in range(heatmaps_shape[0]):
            for waypoint in range(heatmaps_shape[1]):
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1])
                ax.axis("off")
                ax.imshow(
                    rescaled_heatmaps[command, waypoint, ...],
                    cmap="inferno",
                    interpolation="nearest",
                )
                pdf.attach_note(f"command {command} heatmap {waypoint}")
                pdf.savefig(fig)
                plt.close()

            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            ax.axis("off")
            ax.imshow(
                np.sum(rescaled_heatmaps[command, ...], axis=0),
                cmap="inferno",
                interpolation="nearest",
            )
            pdf.attach_note(f"command {command} combined heatmap")
            pdf.savefig()
            plt.close()

        d = pdf.infodict()
        d["Title"] = f"intervention-learning tick {tick}"
        d["Author"] = "intervention-learning"
