import os  # Ensure this is present for os.path
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from skimage.transform import resize
import rasterio  # Add this line
from config import INPUT_FILE, OUTPUT_DIR, SELECTED_BANDS, N_CLUSTERS_MIN, N_CLUSTERS_MAX, MAX_ITERATIONS, MIN_SAMPLES, MAX_STD_DEV, MIN_DIST, MAX_MERGE_PAIRS, CONVERGENCE_THRESHOLD
from utils import save_raster

def perform_classification():
    """Perform IsoData classification on selected bands."""
    print("Classification non-supervisée avec IsoData...")
    print(f"Utilisation de {len(SELECTED_BANDS)} bandes: {SELECTED_BANDS}")

    with rasterio.open(INPUT_FILE) as src:
        bands_data = []
        band_names = []
        for i in SELECTED_BANDS:
            band = src.read(i)
            name = src.descriptions[i-1] if i-1 < len(src.descriptions) else f"Bande {i}"
            bands_data.append(band)
            band_names.append(name)
            print(f"  ✓ Bande {i}: {name} chargée")

        reference_shape = bands_data[0].shape
        for i, band in enumerate(bands_data):
            if band.shape != reference_shape:
                print(f"  Redimensionnement de '{band_names[i]}' à {reference_shape}")
                bands_data[i] = resize(band, reference_shape, preserve_range=True)

        stack = np.stack(bands_data)
        n_bands, height, width = stack.shape
        print(f"Stack de dimensions: {n_bands} bandes x {height} lignes x {width} colonnes")

        data_for_clustering = stack.reshape(n_bands, -1).T
        valid_pixels = ~np.isnan(data_for_clustering).any(axis=1)
        valid_data_raw = data_for_clustering[valid_pixels]

        scaler = StandardScaler()
        valid_data = scaler.fit_transform(valid_data_raw)
        print(f"Données normalisées: {valid_data.shape[0]} pixels valides avec {n_bands} dimensions")

        print(f"Initialisation avec {N_CLUSTERS_MIN} classes...")
        kmeans_init = KMeans(n_clusters=N_CLUSTERS_MIN, max_iter=10, init='k-means++', random_state=42, n_init='auto')
        labels = kmeans_init.fit_predict(valid_data)
        centers = kmeans_init.cluster_centers_

        print("Exécution de l'algorithme IsoData...")
        iteration = 0
        n_clusters_current = N_CLUSTERS_MIN
        previous_labels = None
        converged = False

        while iteration < MAX_ITERATIONS and not converged:
            iteration += 1
            distances = np.sqrt(((valid_data[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
            labels = np.argmin(distances, axis=1)

            if previous_labels is not None:
                changes = np.sum(previous_labels != labels)
                change_ratio = changes / len(labels)
                if change_ratio < CONVERGENCE_THRESHOLD:
                    print(f"Convergence atteinte à l'itération {iteration} ({change_ratio:.4f} < {CONVERGENCE_THRESHOLD})")
                    converged = True

            previous_labels = labels.copy()
            unique_labels, counts = np.unique(labels, return_counts=True)
            empty_clusters = np.setdiff1d(np.arange(n_clusters_current), unique_labels)

            for empty in empty_clusters:
                largest_cluster = unique_labels[np.argmax(counts)]
                largest_indices = np.where(labels == largest_cluster)[0]
                largest_points = valid_data[largest_indices]
                center_dist = np.sqrt(((largest_points - centers[largest_cluster]) ** 2).sum(axis=1))
                farthest_point = largest_indices[np.argmax(center_dist)]
                centers[empty] = valid_data[farthest_point]

            for i in range(n_clusters_current):
                if i in unique_labels:
                    cluster_points = valid_data[labels == i]
                    centers[i] = np.mean(cluster_points, axis=0)

            if n_clusters_current < N_CLUSTERS_MAX:
                clusters_to_split = []
                for i in range(n_clusters_current):
                    if i not in unique_labels or counts[list(unique_labels).index(i)] < MIN_SAMPLES:
                        continue
                    cluster_points = valid_data[labels == i]
                    std_devs = np.std(cluster_points, axis=0)
                    if np.any(std_devs > MAX_STD_DEV) and n_clusters_current < N_CLUSTERS_MAX:
                        clusters_to_split.append((i, std_devs))

                clusters_to_split.sort(key=lambda x: np.max(x[1]), reverse=True)
                for i, std_devs in clusters_to_split:
                    if n_clusters_current >= N_CLUSTERS_MAX:
                        break
                    max_var_axis = np.argmax(std_devs)
                    std_dev = std_devs[max_var_axis]
                    centers = np.vstack([centers, centers[i] + np.array([0.5 * std_dev if j == max_var_axis else 0 for j in range(n_bands)])])
                    centers[i] = centers[i] - np.array([0.5 * std_dev if j == max_var_axis else 0 for j in range(n_bands)])
                    n_clusters_current += 1
                    print(f"  Cluster {i} divisé selon l'axe {max_var_axis} (std={std_dev:.4f}) → {n_clusters_current} clusters")

            if n_clusters_current > N_CLUSTERS_MIN:
                center_distances = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
                np.fill_diagonal(center_distances, np.inf)
                merge_count = 0
                while merge_count < MAX_MERGE_PAIRS and n_clusters_current > N_CLUSTERS_MIN:
                    min_dist_idx = np.unravel_index(np.argmin(center_distances), center_distances.shape)
                    min_dist_value = center_distances[min_dist_idx]
                    if min_dist_value > MIN_DIST:
                        break
                    i, j = min_dist_idx
                    weights = np.array([counts[list(unique_labels).index(i)] if i in unique_labels else 0,
                                       counts[list(unique_labels).index(j)] if j in unique_labels else 0])
                    if np.sum(weights) > 0:
                        centers[i] = (centers[i] * weights[0] + centers[j] * weights[1]) / np.sum(weights)
                    centers = np.delete(centers, j, axis=0)
                    labels[labels == j] = i
                    labels[labels > j] -= 1
                    n_clusters_current -= 1
                    print(f"  Clusters {i} et {j} fusionnés (distance={min_dist_value:.4f}) → {n_clusters_current} clusters")
                    center_distances = np.sqrt(((centers[:, np.newaxis, :] - centers[np.newaxis, :, :]) ** 2).sum(axis=2))
                    np.fill_diagonal(center_distances, np.inf)
                    merge_count += 1

            print(f"Itération {iteration}: {n_clusters_current} clusters")

        clusters = labels
        classification = np.zeros(height * width, dtype=np.uint8)
        classification[valid_pixels] = clusters + 1
        classification = classification.reshape(height, width)

        class_path = os.path.join(OUTPUT_DIR, "classification_isodata.tif")
        profile_out = src.profile.copy()
        profile_out.update(count=1, dtype=rasterio.uint8, nodata=0)
        save_raster(classification, profile_out, class_path)

        print(f"\nClassification terminée avec {n_clusters_current} classes")
        unique_classes, class_counts = np.unique(classification, return_counts=True)
        print("\nStatistiques des classes:")
        for i, (cls, count) in enumerate(zip(unique_classes[1:], class_counts[1:])):
            percentage = (count / np.sum(class_counts[1:])) * 100
            print(f"  Classe {cls}: {count} pixels ({percentage:.2f}%)")

        print("\nCréation d'une version avec fusion des classes 4 et 7 (bâtiments)...")
        classification_merged = classification.copy()
        classification_merged[classification == 7] = 4
        class_merged_path = os.path.join(OUTPUT_DIR, "classification_isodata_landcover.tif")
        save_raster(classification_merged, profile_out, class_merged_path)

        unique_classes_merged, class_counts_merged = np.unique(classification_merged, return_counts=True)
        print("\nStatistiques des classes après fusion des bâtiments:")
        for i, (cls, count) in enumerate(zip(unique_classes_merged[1:], class_counts_merged[1:])):
            percentage = (count / np.sum(class_counts_merged[1:])) * 100
            class_name = f"{cls}" + " (bâtiments fusionnés)" if cls == 4 else f"{cls}"
            print(f"  Classe {class_name}: {count} pixels ({percentage:.2f}%)")

    return classification_merged