import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from config import INPUT_FILE, OUTPUT_DIR, USE_SPECTRAL_INDICES, SELECTED_INDICES

def visualize_classification(classification):
    """
    Visualize classification results with a custom colormap.
    
    Parameters:
    -----------
    classification : numpy.ndarray
        Classification array with class values
    """
    print("\n=== Visualisation de la classification ===")
    
    # Define a colormap for visualization
    # Class 0 is typically no data/masked pixels
    class_colors = [
        (0.0, 0.0, 0.0, 0.0),  # Class 0: Transparent (no data)
        (0.2, 0.8, 0.2, 1.0),  # Class 1: Green (Vegetation)
        (0.8, 0.7, 0.4, 1.0),  # Class 2: Tan (Soil/bare earth)
        (0.0, 0.0, 0.8, 1.0),  # Class 3: Blue (Water)
        (0.8, 0.0, 0.0, 1.0),  # Class 4: Red (Buildings/urban)
        (0.6, 0.3, 0.0, 1.0),  # Class 5: Brown (Mixed urban)
        (1.0, 1.0, 0.0, 1.0),  # Class 6: Yellow (Agriculture)
        (0.5, 0.0, 0.5, 1.0),  # Class 7: Purple (Additional buildings)
        (1.0, 0.5, 0.0, 1.0),  # Class 8: Orange (Other)
        (0.7, 0.7, 0.7, 1.0),  # Class 9: Gray (Other)
    ]
    
    # Create a colormap with these colors
    cmap = colors.ListedColormap(class_colors)
    
    # Set up figure and axes
    plt.figure(figsize=(12, 10))
    
    # Plot the classification
    plt.imshow(classification, cmap=cmap, interpolation='none')
    
    # Create color patches for the legend
    legend_elements = []
    unique_classes = np.unique(classification)
    class_names = {
        0: "No Data",
        1: "Vegetation",
        2: "Sol nu",
        3: "Eau",
        4: "Bâtiments",
        5: "Mixte urbain",
        6: "Agriculture",
        7: "Bâtiments supplémentaires",
        8: "Autre 1",
        9: "Autre 2"
    }
    
    for cls in unique_classes:
        if cls == 0:
            continue  # Skip no data class
        color = class_colors[cls]
        legend_elements.append(plt.Rectangle((0, 0), 1, 1, color=color, label=f"Classe {cls}: {class_names.get(cls, 'Non défini')}"))
    
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=10)
    plt.title("Classification ISODATA", fontsize=14)
    plt.axis('off')
    
    # Save the classification visualization
    output_path = os.path.join(OUTPUT_DIR, "classification_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Classification map saved to: {output_path}")
    plt.close()

def visualize_indices(indices_list=None):
    """
    Visualize spectral indices.
    
    Parameters:
    -----------
    indices_list : list
        List of indices to visualize. If None, all available indices from SELECTED_INDICES will be used.
    """
    if not USE_SPECTRAL_INDICES:
        print("Spectral indices visualization is disabled.")
        return
    
    if indices_list is None:
        indices_list = SELECTED_INDICES
    
    print("\n=== Visualisation des indices spectraux ===")
    
    # Load each index from file
    for index_name in indices_list:
        index_file = os.path.join(OUTPUT_DIR, f"{index_name}.tif")
        
        if not os.path.exists(index_file):
            print(f"Index file not found: {index_file}")
            continue
        
        try:
            with rasterio.open(index_file) as src:
                index_data = src.read(1)
                
                # Create a suitable colormap based on the index type
                if index_name.lower() == 'ndvi':
                    # NDVI colormap: Brown to green
                    cmap = create_ndvi_colormap()
                    vmin, vmax = -0.5, 0.9
                    title = "NDVI (Normalized Difference Vegetation Index)"
                    description = "Red to green: Low to high vegetation density"
                
                elif index_name.lower() == 'ndwi' or index_name.lower() == 'mndwi':
                    # Water indices colormap: Brown to blue
                    cmap = create_water_colormap()
                    vmin, vmax = -0.5, 0.8
                    title = f"{index_name.upper()} (Water Index)"
                    description = "Brown to blue: Low to high water content"
                
                elif index_name.lower() == 'ndbi':
                    # Building index colormap: Green to red
                    cmap = create_building_colormap()
                    vmin, vmax = -0.5, 0.6
                    title = "NDBI (Normalized Difference Built-up Index)"
                    description = "Green to red: Low to high built-up density"
                
                elif index_name.lower() == 'bai':
                    # Burned area colormap: Green to purple
                    cmap = plt.cm.RdPu
                    vmin, vmax = np.nanpercentile(index_data, [2, 98])
                    title = "BAI (Burned Area Index)"
                    description = "Light to dark purple: Increasing burn severity"
                
                elif index_name.lower() == 'nbr':
                    # NBR colormap: Red to green
                    cmap = plt.cm.RdYlGn
                    vmin, vmax = -0.5, 0.8
                    title = "NBR (Normalized Burn Ratio)"
                    description = "Red to green: High to low burn severity"
                
                else:
                    # Default colormap
                    cmap = plt.cm.viridis
                    vmin, vmax = np.nanpercentile(index_data, [2, 98])
                    title = f"{index_name.upper()} Index"
                    description = "Blue to yellow: Low to high values"
                
                # Replace extreme values and NaN
                index_data_cleaned = np.clip(index_data, vmin, vmax)
                index_data_cleaned = np.nan_to_num(index_data_cleaned, nan=vmin)
                
                # Plot the index
                plt.figure(figsize=(12, 10))
                img = plt.imshow(index_data_cleaned, cmap=cmap, vmin=vmin, vmax=vmax)
                plt.colorbar(img, label=description, shrink=0.8)
                plt.title(title, fontsize=14)
                plt.axis('off')
                
                # Save the visualization
                output_path = os.path.join(OUTPUT_DIR, f"{index_name}_map.png")
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                print(f"{index_name.upper()} visualization saved to: {output_path}")
                plt.close()
        
        except Exception as e:
            print(f"Error visualizing {index_name}: {str(e)}")

def visualize_wetland_mask():
    """Visualize the wetland mask."""
    wetland_file = os.path.join(OUTPUT_DIR, "wetland_mask.tif")
    
    if not os.path.exists(wetland_file):
        print(f"Wetland mask file not found: {wetland_file}")
        return
    
    try:
        with rasterio.open(wetland_file) as src:
            mask_data = src.read(1)
            
            # Create a custom colormap for the mask
            colors_mask = [(1, 1, 1, 0), (0, 0.5, 1, 0.7)]  # Transparent to blue
            cmap_mask = LinearSegmentedColormap.from_list('wetland_mask', colors_mask, N=2)
            
            # Plot the mask
            plt.figure(figsize=(12, 10))
            plt.imshow(mask_data, cmap=cmap_mask, interpolation='none')
            plt.title("Wetland Mask", fontsize=14)
            plt.axis('off')
            
            # Save the visualization
            output_path = os.path.join(OUTPUT_DIR, "wetland_mask_map.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Wetland mask visualization saved to: {output_path}")
            plt.close()
    
    except Exception as e:
        print(f"Error visualizing wetland mask: {str(e)}")

def visualize_building_mask():
    """Visualize the building mask."""
    building_file = os.path.join(OUTPUT_DIR, "building_mask.tif")
    
    if not os.path.exists(building_file):
        print(f"Building mask file not found: {building_file}")
        return
    
    try:
        with rasterio.open(building_file) as src:
            mask_data = src.read(1)
            
            # Create a custom colormap for the mask
            colors_mask = [(1, 1, 1, 0), (0.8, 0, 0, 0.7)]  # Transparent to red
            cmap_mask = LinearSegmentedColormap.from_list('building_mask', colors_mask, N=2)
            
            # Plot the mask
            plt.figure(figsize=(12, 10))
            plt.imshow(mask_data, cmap=cmap_mask, interpolation='none')
            plt.title("Building/Urban Mask", fontsize=14)
            plt.axis('off')
            
            # Save the visualization
            output_path = os.path.join(OUTPUT_DIR, "building_mask_map.png")
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Building mask visualization saved to: {output_path}")
            plt.close()
    
    except Exception as e:
        print(f"Error visualizing building mask: {str(e)}")

def create_ndvi_colormap():
    """Create a custom colormap for NDVI visualization."""
    colors_ndvi = [
        (0.6, 0.4, 0.2),  # Brown for low values
        (0.9, 0.9, 0.3),  # Yellow
        (0.3, 0.6, 0.3),  # Light green
        (0.0, 0.4, 0.0)   # Dark green for high values
    ]
    return LinearSegmentedColormap.from_list('ndvi_cmap', colors_ndvi)

def create_water_colormap():
    """Create a custom colormap for water indices visualization."""
    colors_water = [
        (0.6, 0.4, 0.2),  # Brown for low values
        (0.9, 0.9, 0.6),  # Light brown/tan
        (0.6, 0.8, 0.9),  # Light blue
        (0.0, 0.0, 0.6)   # Dark blue for high values
    ]
    return LinearSegmentedColormap.from_list('water_cmap', colors_water)

def create_building_colormap():
    """Create a custom colormap for building indices visualization."""
    colors_building = [
        (0.0, 0.4, 0.0),  # Dark green for low values
        (0.8, 0.8, 0.4),  # Light yellow
        (0.9, 0.6, 0.3),  # Orange
        (0.6, 0.0, 0.0)   # Dark red for high values
    ]
    return LinearSegmentedColormap.from_list('building_cmap', colors_building)

def visualize_all():
    """Visualize all available indices and masks."""
    if USE_SPECTRAL_INDICES:
        visualize_indices()
        visualize_wetland_mask()
        visualize_building_mask()
