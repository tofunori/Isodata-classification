import numpy as np
import rasterio
import os
from config import INPUT_FILE, OUTPUT_DIR, BAND_MAPPING

def calculate_indices(indices_list=None, save_outputs=True):
    """
    Calculate spectral indices from multispectral imagery.
    
    Parameters:
    -----------
    indices_list : list
        List of indices to calculate. If None, all available indices will be calculated.
        Available indices: 'ndvi', 'ndwi', 'ndbi', 'mndwi', 'bai', 'nbr'
    save_outputs : bool
        Whether to save the indices as GeoTIFF files
        
    Returns:
    --------
    dict
        Dictionary containing the calculated indices as numpy arrays
    """
    print("Calculating spectral indices...")
    
    # Define all available indices
    available_indices = ['ndvi', 'ndwi', 'ndbi', 'mndwi', 'bai', 'nbr']
    
    # If no indices specified, calculate all
    if indices_list is None:
        indices_list = available_indices
    else:
        # Validate requested indices
        for idx in indices_list:
            if idx.lower() not in available_indices:
                print(f"Warning: Index '{idx}' is not recognized and will be skipped.")
        indices_list = [idx.lower() for idx in indices_list if idx.lower() in available_indices]
    
    # Dictionary to store band data
    bands = {}
    
    # Dictionary to store calculated indices
    indices = {}
    
    print(f"Opening raster file: {INPUT_FILE}")
    with rasterio.open(INPUT_FILE) as src:
        # Load required bands based on the requested indices
        required_bands = set()
        
        if 'ndvi' in indices_list:
            required_bands.update(['red', 'nir'])
        
        if 'ndwi' in indices_list:
            required_bands.update(['green', 'nir'])
        
        if 'mndwi' in indices_list:
            required_bands.update(['green', 'swir1'])
        
        if 'ndbi' in indices_list:
            required_bands.update(['nir', 'swir1'])
        
        if 'bai' in indices_list:
            required_bands.update(['red', 'nir'])
        
        if 'nbr' in indices_list:
            required_bands.update(['nir', 'swir2'])
        
        # Load the required bands
        for band_name in required_bands:
            band_number = BAND_MAPPING.get(band_name)
            if band_number is None:
                print(f"Warning: Band mapping for '{band_name}' not found.")
                continue
                
            if band_number > src.count:
                print(f"Warning: Band {band_number} requested but input file only has {src.count} bands.")
                continue
                
            print(f"  Loading {band_name.upper()} band (Band {band_number})...")
            bands[band_name] = src.read(band_number).astype(np.float32)
            
            # Convert to reflectance if necessary (assuming the data is already in reflectance units)
            # If your data is in DN values, you would perform conversion here
            
            # Handle invalid values
            bands[band_name] = np.where(bands[band_name] <= 0, np.nan, bands[band_name])
        
        # Get metadata for output
        profile = src.profile.copy()
        profile.update(count=1, dtype='float32', nodata=np.nan)
        
        # Calculate the requested indices
        print("\nCalculating indices:")
        
        # NDVI (Normalized Difference Vegetation Index)
        if 'ndvi' in indices_list and 'red' in bands and 'nir' in bands:
            print("  Calculating NDVI (Normalized Difference Vegetation Index)...")
            ndvi = (bands['nir'] - bands['red']) / (bands['nir'] + bands['red'])
            indices['ndvi'] = ndvi
            
            if save_outputs:
                ndvi_path = os.path.join(OUTPUT_DIR, "ndvi.tif")
                with rasterio.open(ndvi_path, 'w', **profile) as dst:
                    dst.write(ndvi, 1)
                print(f"    NDVI saved to: {ndvi_path}")
        
        # NDWI (Normalized Difference Water Index) - For water bodies
        if 'ndwi' in indices_list and 'green' in bands and 'nir' in bands:
            print("  Calculating NDWI (Normalized Difference Water Index)...")
            ndwi = (bands['green'] - bands['nir']) / (bands['green'] + bands['nir'])
            indices['ndwi'] = ndwi
            
            if save_outputs:
                ndwi_path = os.path.join(OUTPUT_DIR, "ndwi.tif")
                with rasterio.open(ndwi_path, 'w', **profile) as dst:
                    dst.write(ndwi, 1)
                print(f"    NDWI saved to: {ndwi_path}")
        
        # MNDWI (Modified Normalized Difference Water Index) - Better for turbid water
        if 'mndwi' in indices_list and 'green' in bands and 'swir1' in bands:
            print("  Calculating MNDWI (Modified Normalized Difference Water Index)...")
            mndwi = (bands['green'] - bands['swir1']) / (bands['green'] + bands['swir1'])
            indices['mndwi'] = mndwi
            
            if save_outputs:
                mndwi_path = os.path.join(OUTPUT_DIR, "mndwi.tif")
                with rasterio.open(mndwi_path, 'w', **profile) as dst:
                    dst.write(mndwi, 1)
                print(f"    MNDWI saved to: {mndwi_path}")
        
        # NDBI (Normalized Difference Built-up Index) - For urban/built-up areas
        if 'ndbi' in indices_list and 'nir' in bands and 'swir1' in bands:
            print("  Calculating NDBI (Normalized Difference Built-up Index)...")
            ndbi = (bands['swir1'] - bands['nir']) / (bands['swir1'] + bands['nir'])
            indices['ndbi'] = ndbi
            
            if save_outputs:
                ndbi_path = os.path.join(OUTPUT_DIR, "ndbi.tif")
                with rasterio.open(ndbi_path, 'w', **profile) as dst:
                    dst.write(ndbi, 1)
                print(f"    NDBI saved to: {ndbi_path}")
        
        # BAI (Burned Area Index) - For burned areas
        if 'bai' in indices_list and 'red' in bands and 'nir' in bands:
            print("  Calculating BAI (Burned Area Index)...")
            # Constants for BAI
            pc_red = 0.1  # Reference value for charred vegetation in RED
            pc_nir = 0.06  # Reference value for charred vegetation in NIR
            bai = 1.0 / ((0.1 - bands['red'])**2 + (0.06 - bands['nir'])**2)
            indices['bai'] = bai
            
            if save_outputs:
                bai_path = os.path.join(OUTPUT_DIR, "bai.tif")
                with rasterio.open(bai_path, 'w', **profile) as dst:
                    dst.write(bai, 1)
                print(f"    BAI saved to: {bai_path}")
        
        # NBR (Normalized Burn Ratio) - For fire severity
        if 'nbr' in indices_list and 'nir' in bands and 'swir2' in bands:
            print("  Calculating NBR (Normalized Burn Ratio)...")
            nbr = (bands['nir'] - bands['swir2']) / (bands['nir'] + bands['swir2'])
            indices['nbr'] = nbr
            
            if save_outputs:
                nbr_path = os.path.join(OUTPUT_DIR, "nbr.tif")
                with rasterio.open(nbr_path, 'w', **profile) as dst:
                    dst.write(nbr, 1)
                print(f"    NBR saved to: {nbr_path}")
    
    # Report completion
    print(f"\nCalculated {len(indices)} spectral indices: {', '.join(indices.keys())}")
    
    return indices

def create_wetland_mask(ndwi_threshold=0.3, mndwi_threshold=0.2, save_mask=True):
    """
    Create a binary mask for potential wetland areas by combining NDWI and MNDWI.
    
    Parameters:
    -----------
    ndwi_threshold : float
        Threshold value for NDWI (higher values indicate more likely water)
    mndwi_threshold : float
        Threshold value for MNDWI (higher values indicate more likely water)
    save_mask : bool
        Whether to save the wetland mask as a GeoTIFF file
        
    Returns:
    --------
    numpy.ndarray
        Binary mask where 1 = potential wetland, 0 = non-wetland
    """
    print("\nCreating wetland mask...")
    
    # Calculate the required indices
    indices = calculate_indices(indices_list=['ndwi', 'mndwi'], save_outputs=False)
    
    if 'ndwi' not in indices or 'mndwi' not in indices:
        print("Error: Could not calculate required indices for wetland mask.")
        return None
    
    # Create masks for each index
    ndwi_mask = indices['ndwi'] > ndwi_threshold
    mndwi_mask = indices['mndwi'] > mndwi_threshold
    
    # Combine masks - potential wetland where either NDWI or MNDWI indicate water
    wetland_mask = np.logical_or(ndwi_mask, mndwi_mask).astype(np.uint8)
    
    if save_mask:
        with rasterio.open(INPUT_FILE) as src:
            profile = src.profile.copy()
            profile.update(count=1, dtype='uint8', nodata=255)
            
            mask_path = os.path.join(OUTPUT_DIR, "wetland_mask.tif")
            with rasterio.open(mask_path, 'w', **profile) as dst:
                dst.write(wetland_mask, 1)
            print(f"Wetland mask saved to: {mask_path}")
    
    print(f"Wetland mask created. Potential wetland area: {np.sum(wetland_mask)} pixels")
    return wetland_mask

def create_building_mask(ndbi_threshold=0.1, save_mask=True):
    """
    Create a binary mask for potential building/urban areas using NDBI.
    
    Parameters:
    -----------
    ndbi_threshold : float
        Threshold value for NDBI (higher values indicate more likely built-up areas)
    save_mask : bool
        Whether to save the building mask as a GeoTIFF file
        
    Returns:
    --------
    numpy.ndarray
        Binary mask where 1 = potential building, 0 = non-building
    """
    print("\nCreating building/urban mask...")
    
    # Calculate the required indices
    indices = calculate_indices(indices_list=['ndbi'], save_outputs=False)
    
    if 'ndbi' not in indices:
        print("Error: Could not calculate NDBI for building mask.")
        return None
    
    # Create mask for built-up areas
    building_mask = (indices['ndbi'] > ndbi_threshold).astype(np.uint8)
    
    if save_mask:
        with rasterio.open(INPUT_FILE) as src:
            profile = src.profile.copy()
            profile.update(count=1, dtype='uint8', nodata=255)
            
            mask_path = os.path.join(OUTPUT_DIR, "building_mask.tif")
            with rasterio.open(mask_path, 'w', **profile) as dst:
                dst.write(building_mask, 1)
            print(f"Building/urban mask saved to: {mask_path}")
    
    print(f"Building/urban mask created. Potential built-up area: {np.sum(building_mask)} pixels")
    return building_mask

def stack_indices_for_classification(indices_list=None):
    """
    Calculate and stack selected indices as additional bands for classification.
    
    Parameters:
    -----------
    indices_list : list
        List of indices to include. If None, all indices will be calculated.
        
    Returns:
    --------
    numpy.ndarray
        Array of stacked indices (n_indices x height x width)
    list
        List of index names in the same order as the stacked array
    """
    print("\nPreparing spectral indices for classification...")
    
    # Calculate all requested indices
    indices_dict = calculate_indices(indices_list=indices_list, save_outputs=False)
    
    if not indices_dict:
        print("No indices could be calculated.")
        return None, []
    
    # Get list of indices and their arrays
    index_names = list(indices_dict.keys())
    index_arrays = [indices_dict[name] for name in index_names]
    
    # Stack the arrays
    stacked_indices = np.stack(index_arrays)
    
    print(f"Prepared stack with {len(index_names)} indices: {', '.join(index_names)}")
    print(f"Stack dimensions: {stacked_indices.shape}")
    
    return stacked_indices, index_names

if __name__ == "__main__":
    # Example: Calculate all available indices
    indices = calculate_indices()
    
    # Example: Create wetland and building masks
    wetland_mask = create_wetland_mask()
    building_mask = create_building_mask()
    
    # Example: Prepare indices for classification
    stacked_indices, index_names = stack_indices_for_classification(['ndvi', 'ndwi', 'ndbi'])
