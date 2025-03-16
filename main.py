import rasterio
import numpy as np
from band_visualization import visualize_bands
from classification import perform_classification
from visualization import visualize_classification
from basic_statistics import compute_basic_statistics
from advanced_statistics import compute_advanced_statistics
from utils import get_image_info
from config import INPUT_FILE, USE_SPECTRAL_INDICES, SELECTED_INDICES
from spectral_indices import calculate_indices, stack_indices_for_classification
import os

def main():
    """Main function to run the IsoData classification workflow."""
    # Step 1: Display image info
    with rasterio.open(INPUT_FILE) as src:
        get_image_info(src)

    # Step 2: Visualize bands
    visualize_bands()
    
    # Step 3: Calculate spectral indices if enabled
    if USE_SPECTRAL_INDICES:
        print(f"\n=== Calculating Spectral Indices ===")
        # Calculate all requested indices for visualization
        calculate_indices(indices_list=SELECTED_INDICES)
        
        # Stack indices for classification
        stacked_indices, index_names = stack_indices_for_classification(SELECTED_INDICES)
        print(f"Indices prepared for classification: {', '.join(index_names)}")
    else:
        stacked_indices = None
        index_names = []

    # Step 4: Perform classification
    classification = perform_classification(spectral_indices=stacked_indices, 
                                           index_names=index_names)

    # Step 5: Visualize classification results
    visualize_classification(classification)

    # Step 6: Compute basic statistics
    compute_basic_statistics(classification)

    # Step 7: Compute advanced statistics
    compute_advanced_statistics(classification)

if __name__ == "__main__":
    main()
