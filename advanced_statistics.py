import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
import os
from scipy.spatial import distance
from scipy.stats import entropy
from scipy.ndimage import label, generate_binary_structure
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from config import INPUT_FILE, OUTPUT_DIR, SELECTED_BANDS

def compute_advanced_statistics(classification):
    """Compute advanced statistics for the classification."""
    print("Calcul des statistiques complémentaires...")
    with rasterio.open(INPUT_FILE) as src:
        unique_classes = np.unique(classification)
        class_ids = [c for c in unique_classes if c > 0]
        n_classes = len(class_ids)

        # 1. Separability Analysis
        centroids = {}
        for class_id in class_ids:
            class_mask = (classification == class_id)
            class_data = np.array([src.read(band_idx)[class_mask] for band_idx in SELECTED_BANDS])
            centroids[class_id] = np.nanmean(class_data, axis=1)

        distance_matrix = np.zeros((n_classes, n_classes))
        for i, class1 in enumerate(class_ids):
            for j, class2 in enumerate(class_ids):
                if i != j:
                    distance_matrix[i, j] = distance.euclidean(centroids[class1], centroids[class2])

        distance_df = pd.DataFrame(distance_matrix, index=[f'Classe {c}' for c in class_ids], columns=[f'Classe {c}' for c in class_ids])
        print("\n1. Analyse de la séparabilité des classes")
        print("Matrice de distances euclidiennes entre centroïdes des classes:")
        print(distance_df.to_string())

        plt.figure(figsize=(10, 8))
        sns.heatmap(distance_df, annot=True, cmap='viridis', fmt='.2f')
        plt.title('Distances euclidiennes entre les centroïdes des classes')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "distances_centroids.png"), dpi=300)
        plt.show()

        min_distances = [np.min(distance_matrix[i, :][distance_matrix[i, :] > 0]) for i in range(n_classes) if len(distance_matrix[i, :][distance_matrix[i, :] > 0]) > 0]
        print(f"Distance minimale entre deux classes: {np.min(min_distances):.4f}")
        print(f"Distance maximale entre deux classes: {np.max(distance_matrix):.4f}")
        print(f"Distance moyenne entre les classes: {np.mean(distance_matrix[distance_matrix > 0]):.4f}")
        min_dist_idx = np.where(distance_matrix == np.min(min_distances))
        print(f"Classes les plus similaires: Classe {class_ids[min_dist_idx[0][0]]} et Classe {class_ids[min_dist_idx[1][0]]}")
        max_dist_idx = np.where(distance_matrix == np.max(distance_matrix))
        print(f"Classes les plus différentes: Classe {class_ids[max_dist_idx[0][0]]} et Classe {class_ids[max_dist_idx[1][0]]}")

        # 2. Variance Analysis
        print("\n2. Analyse de la variance intra-classe vs inter-classe")
        intra_class_variance = {}
        for class_id in class_ids:
            class_mask = (classification == class_id)
            class_data = np.array([src.read(band_idx)[class_mask] for band_idx in SELECTED_BANDS])
            class_data = class_data.reshape(len(SELECTED_BANDS), -1).T
            var_by_band = np.nanvar(class_data, axis=0)
            intra_class_variance[class_id] = np.sum(var_by_band)

        var_df = pd.DataFrame({'Classe': list(intra_class_variance.keys()), 'Variance intra-classe': list(intra_class_variance.values())})
        var_df = var_df.sort_values('Variance intra-classe', ascending=False)
        print("Variance intra-classe par classe:")
        print(var_df.to_string(index=False))

        plt.figure(figsize=(10, 6))
        plt.bar(var_df['Classe'].astype(str), var_df['Variance intra-classe'], color='teal')
        plt.xlabel('Classe')
        plt.ylabel('Variance intra-classe')
        plt.title('Variance intra-classe par classe')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "variance_intra_classe.png"), dpi=300)
        plt.show()

        fisher_ratios = []
        class_pairs = []
        for i, class1 in enumerate(class_ids):
            for j, class2 in enumerate(class_ids):
                if i < j:
                    inter_var = np.sum((centroids[class1] - centroids[class2])**2)
                    intra_var = intra_class_variance[class1] + intra_class_variance[class2]
                    if intra_var > 0:
                        fisher_ratios.append(inter_var / intra_var)
                        class_pairs.append((class1, class2))

        fisher_df = pd.DataFrame({'Classe 1': [pair[0] for pair in class_pairs], 'Classe 2': [pair[1] for pair in class_pairs], 'Ratio de Fisher': fisher_ratios})
        fisher_df = fisher_df.sort_values('Ratio de Fisher', ascending=False)
        print("\nRatio de Fisher (variance inter-classe / variance intra-classe):")
        print(fisher_df.head(10).to_string(index=False))
        print(f"Ratio de Fisher moyen: {np.mean(fisher_ratios):.4f}")
        print(f"Ratio de Fisher médian: {np.median(fisher_ratios):.4f}")

        # 3. Spatial Analysis
        print("\n3. Analyse spatiale des clusters")
        structure = generate_binary_structure(2, 2)
        spatial_stats = []
        for class_id in class_ids:
            class_mask = (classification == class_id).astype(np.int32)
            labeled_array, num_features = label(class_mask, structure=structure)
            component_sizes = np.bincount(labeled_array.flatten())[1:]
            spatial_stats.append({
                'Classe': class_id, 'Nombre de pixels': np.sum(class_mask), 'Nombre de régions': num_features,
                'Taille moyenne des régions': np.mean(component_sizes) if num_features > 0 else 0,
                'Taille médiane des régions': np.median(component_sizes) if num_features > 0 else 0,
                'Plus grande région (pixels)': np.max(component_sizes) if num_features > 0 else 0,
                'Plus petite région (pixels)': np.min(component_sizes) if num_features > 0 else 0,
                'Fragmentation (régions/pixels)': num_features / np.sum(class_mask) if np.sum(class_mask) > 0 else 0
            })

        spatial_df = pd.DataFrame(spatial_stats).sort_values('Nombre de pixels', ascending=False)
        print("Statistiques spatiales par classe:")
        print(spatial_df.to_string(index=False))

        plt.figure(figsize=(12, 6))
        plt.bar(spatial_df['Classe'].astype(str), spatial_df['Fragmentation (régions/pixels)'], color='purple')
        plt.xlabel('Classe')
        plt.ylabel('Indice de fragmentation (régions/pixels)')
        plt.title('Fragmentation spatiale par classe')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "fragmentation_spatiale.png"), dpi=300)
        plt.show()

        # 4. PCA Visualization
        print("\n4. Visualisation des classes dans l'espace ACP")
        data_for_pca = []
        labels_for_pca = []
        max_samples_per_class = 5000
        for class_id in class_ids:
            class_mask = (classification == class_id)
            class_data = np.array([src.read(band_idx)[class_mask] for band_idx in SELECTED_BANDS])
            class_data = class_data.reshape(len(SELECTED_BANDS), -1).T
            valid_indices = ~np.isnan(class_data).any(axis=1)
            valid_data = class_data[valid_indices]
            if valid_data.shape[0] > max_samples_per_class:
                sample_indices = np.random.choice(valid_data.shape[0], max_samples_per_class, replace=False)
                valid_data = valid_data[sample_indices]
            data_for_pca.append(valid_data)
            labels_for_pca.extend([class_id] * valid_data.shape[0])

        if data_for_pca:
            all_data = np.vstack(data_for_pca)
            labels = np.array(labels_for_pca)
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(all_data)
            pca_df = pd.DataFrame({'PC1': pca_result[:, 0], 'PC2': pca_result[:, 1], 'Classe': labels})
            explained_variance = pca.explained_variance_ratio_ * 100
            print(f"Variance expliquée: PC1 = {explained_variance[0]:.2f}%, PC2 = {explained_variance[1]:.2f}%")

            plt.figure(figsize=(12, 10))
            colors = plt.cm.tab20(np.linspace(0, 1, n_classes))
            for i, class_id in enumerate(class_ids):
                class_data = pca_df[pca_df['Classe'] == class_id]
                plt.scatter(class_data['PC1'], class_data['PC2'], c=[colors[i]], label=f'Classe {class_id}', alpha=0.7, edgecolors='w', linewidth=0.5)
            plt.xlabel(f'PC1 ({explained_variance[0]:.2f}%)')
            plt.ylabel(f'PC2 ({explained_variance[1]:.2f}%)')
            plt.title("Visualisation des classes dans l'espace ACP")
            plt.legend(loc='best')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "classes_acp.png"), dpi=300)
            plt.show()

            class_separations = []
            for i, class1 in enumerate(class_ids):
                for j, class2 in enumerate(class_ids):
                    if i < j:
                        data1 = pca_df[pca_df['Classe'] == class1][['PC1', 'PC2']].values
                        data2 = pca_df[pca_df['Classe'] == class2][['PC1', 'PC2']].values
                        if len(data1) > 0 and len(data2) > 0:
                            max_points = 1000
                            if len(data1) > max_points:
                                indices = np.random.choice(len(data1), max_points, replace=False)
                                data1 = data1[indices]
                            if len(data2) > max_points:
                                indices = np.random.choice(len(data2), max_points, replace=False)
                                data2 = data2[indices]
                            distances = pairwise_distances(data1, data2)
                            class_separations.append({'Classe 1': class1, 'Classe 2': class2, "Distance moyenne dans l'espace ACP": np.mean(distances)})

            separation_df = pd.DataFrame(class_separations).sort_values("Distance moyenne dans l'espace ACP", ascending=False)
            print("\nSéparation des classes dans l'espace ACP:")
            print(separation_df.to_string(index=False))
        else:
            print("Pas assez de données valides pour effectuer l'ACP.")

        # 5. Entropy Analysis
        print("\n5. Indices d'homogénéité et d'entropie")
        shannon_entropy = []
        for class_id in class_ids:
            class_mask = (classification == class_id)
            band_data = src.read(SELECTED_BANDS[0])[class_mask]
            hist, _ = np.histogram(band_data, bins=50)
            prob = hist / np.sum(hist)
            class_entropy = entropy(prob[prob > 0])
            shannon_entropy.append({'Classe': class_id, 'Entropie de Shannon': class_entropy, 'Nombre de pixels': np.sum(class_mask)})

        entropy_df = pd.DataFrame(shannon_entropy)
        entropy_df['Entropie normalisée'] = entropy_df['Entropie de Shannon'] / np.log(50)
        entropy_df = entropy_df.sort_values('Entropie normalisée', ascending=False)
        print("Entropie de Shannon par classe (mesure d'homogénéité):")
        print(entropy_df.to_string(index=False))

        plt.figure(figsize=(12, 6))
        plt.bar(entropy_df['Classe'].astype(str), entropy_df['Entropie normalisée'], color='orange')
        plt.xlabel('Classe')
        plt.ylabel('Entropie normalisée')
        plt.title('Entropie de Shannon par classe (plus la valeur est basse, plus la classe est homogène)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "entropie_classes.png"), dpi=300)
        plt.show()

        # Export results
        try:
            distance_df.to_csv(os.path.join(OUTPUT_DIR, "distance_matrix.csv"))
            var_df.to_csv(os.path.join(OUTPUT_DIR, "intra_class_variance.csv"), index=False)
            fisher_df.to_csv(os.path.join(OUTPUT_DIR, "fisher_ratios.csv"), index=False)
            spatial_df.to_csv(os.path.join(OUTPUT_DIR, "spatial_statistics.csv"), index=False)
            entropy_df.to_csv(os.path.join(OUTPUT_DIR, "entropy_statistics.csv"), index=False)
            if 'separation_df' in locals():
                separation_df.to_csv(os.path.join(OUTPUT_DIR, "acp_separation.csv"), index=False)
            print(f"\nStatistiques complémentaires exportées dans {OUTPUT_DIR}")
        except Exception as e:
            print(f"Erreur lors de l'exportation des statistiques complémentaires: {e}")

        print("\nAnalyse des statistiques complémentaires terminée.")