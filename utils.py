import os  # Add this line
import rasterio
import numpy as np

def load_band(image_path, band_index):
    """Load a specific band from an image."""
    with rasterio.open(image_path) as src:
        band = src.read(band_index)
        profile = src.profile.copy()
    return band, profile

def save_raster(data, profile, output_path, dtype=None):
    """Save a raster to GeoTIFF format."""
    if dtype:
        profile.update(count=1, dtype=dtype, compress='lzw')
    with rasterio.open(output_path, 'w', **profile) as dst:
        dst.write(data, 1)
    print(f"Fichier sauvegardé: {output_path}")

def get_image_info(src):
    """Print basic image information."""
    print(f"Informations sur l'image {os.path.basename(src.name)}:")
    print(f"Dimensions: {src.width} x {src.height} pixels")
    print(f"Nombre total de bandes: {src.count}")
    print(f"Type de données: {src.dtypes[0]}")
    if src.crs:
        print(f"Système de coordonnées: {src.crs.to_string()}")
    print("\nBandes disponibles:")
    for i in range(1, src.count + 1):
        desc = src.descriptions[i-1] if src.descriptions and i-1 < len(src.descriptions) else "Non spécifiée"
        print(f"  Bande {i}: {desc}")