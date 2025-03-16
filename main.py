import rasterio
from band_visualization import visualize_bands
from classification import perform_classification
from visualization import visualize_classification
from basic_statistics import compute_basic_statistics
from advanced_statistics import compute_advanced_statistics
from utils import get_image_info
from config import INPUT_FILE

def main():
    """Main function to run the IsoData classification workflow."""
    # Step 1: Display image info
    with rasterio.open(INPUT_FILE) as src:
        get_image_info(src)

    # Step 2: Visualize bands
    visualize_bands()

    # Step 3: Perform classification
    classification = perform_classification()

    # Step 4: Visualize classification results
    visualize_classification(classification)

    # Step 5: Compute basic statistics
    compute_basic_statistics(classification)

    # Step 6: Compute advanced statistics
    compute_advanced_statistics(classification)

if __name__ == "__main__":
    main()