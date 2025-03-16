import os  # Add this line
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from config import INPUT_FILE, OUTPUT_DIR, BANDS_TO_EXCLUDE

def visualize_bands():
    """Visualize selected bands from the input raster."""
    print("Visualisation des bandes sélectionnées...")
    with rasterio.open(INPUT_FILE) as src:
        print(f"File opened: {INPUT_FILE}")
        bands_to_display = [i+1 for i in range(src.count) if i+1 not in BANDS_TO_EXCLUDE]
        n_bands_to_display = len(bands_to_display)
        print(f"Affichage de {n_bands_to_display} bandes sur {src.count} (bandes {BANDS_TO_EXCLUDE} exclues)")

        n_cols = min(4, n_bands_to_display)
        n_rows = int(np.ceil(n_bands_to_display / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))

        if n_rows > 1 and n_cols > 1:
            axes = axes.flatten()
        elif n_rows == 1 and n_cols > 1:
            axes = axes
        elif n_cols == 1 and n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]

        for idx, band_num in enumerate(bands_to_display):
            band = src.read(band_num)
            band_name = src.descriptions[band_num-1] if band_num-1 < len(src.descriptions) else f"Bande {band_num}"
            band_display = np.copy(band)
            if src.nodata is not None:
                band_display[band == src.nodata] = np.nan
            valid_data = band_display[~np.isnan(band_display)]
            vmin, vmax = np.nanpercentile(band_display, [2, 98]) if len(valid_data) > 0 else (0, 1)
            im = axes[idx].imshow(band_display, cmap='gray', vmin=vmin, vmax=vmax)
            axes[idx].set_title(f"{band_name}", fontsize=10)
            axes[idx].set_xticks([])
            axes[idx].set_yticks([])
            cbar = plt.colorbar(im, ax=axes[idx], fraction=0.035, pad=0.04, shrink=0.7)
            cbar.ax.tick_params(labelsize=8)
            mean_val = np.nanmean(band_display)
            std_val = np.nanstd(band_display)
            axes[idx].text(0.05, 0.95, f"μ={mean_val:.2f}, σ={std_val:.2f}", 
                           transform=axes[idx].transAxes, fontsize=8,
                           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        for i in range(n_bands_to_display, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        plt.suptitle("Aperçu des bandes spectrales sélectionnées", fontsize=16, y=1.02)
        plt.subplots_adjust(top=0.95)
        plt.savefig(os.path.join(OUTPUT_DIR, "selected_bands_preview.png"), dpi=300, bbox_inches='tight')
        plt.show()
        print("Visualisation des bandes sélectionnées terminée!")

if __name__ == "__main__":
    visualize_bands()