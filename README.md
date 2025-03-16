# Isodata-classification

A Python implementation of the ISODATA (Iterative Self-Organizing Data Analysis Technique) algorithm for unsupervised classification of remote sensing imagery, with additional spectral indices functionality for improved land cover classification.

## Overview

This project implements the ISODATA unsupervised classification algorithm for satellite imagery, with specific focus on land cover classification. It now includes support for spectral indices calculation and visualization, which can significantly improve classification accuracy for specific land cover types like water bodies, vegetation, and built-up areas.

## Features

- Iterative Self-Organizing Data Analysis Technique (ISODATA) implementation
- Spectral indices calculation (NDVI, NDWI, NDBI, MNDWI, BAI, NBR)
- Wetland and building mask generation
- Visualization of bands, indices, and classification results
- Basic and advanced statistical analysis of classification results

## Spectral Indices

The following spectral indices are supported:

- **NDVI** (Normalized Difference Vegetation Index): Highlights vegetation density
- **NDWI** (Normalized Difference Water Index): Highlights water bodies
- **MNDWI** (Modified Normalized Difference Water Index): Better for turbid water
- **NDBI** (Normalized Difference Built-up Index): Highlights urban/built-up areas
- **BAI** (Burned Area Index): Highlights burned areas
- **NBR** (Normalized Burn Ratio): Assesses burn severity

## Configuration

All parameters can be adjusted in the `config.py` file:

- Input and output paths
- Selected bands for classification
- Spectral indices options
- ISODATA parameters (clusters, iterations, thresholds)

## Usage

1. Configure your settings in `config.py`
2. Run the classification:
   ```
   python main.py
   ```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- scikit-learn
- scikit-image
- rasterio

Install dependencies:
```
pip install -r requirements.txt
```

## Workflow

1. Load and display image information
2. Visualize selected bands
3. Calculate spectral indices (NDVI, NDWI, NDBI, etc.)
4. Generate wetland and building masks
5. Perform ISODATA classification
6. Visualize classification results
7. Compute statistics on classification results

## Example Results

The classification produces multiple outputs in the specified output directory:
- Spectral indices as GeoTIFF files
- Wetland and building masks
- Classification map
- Visualizations of bands, indices, and classification
- Statistical reports

## License

This project is open source and available under the MIT License.
