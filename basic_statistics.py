import numpy as np
import pandas as pd
import rasterio
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import StratifiedShuffleSplit
from config import INPUT_FILE, OUTPUT_DIR, SELECTED_BANDS

def compute_basic_statistics(classification):
    """Compute basic statistics for the classification."""
    print("Génération des statistiques de classification...")
    with rasterio.open(INPUT_FILE) as src:
        stack = np.stack([src.read(i) for i in SELECTED_BANDS])
        n_bands, _, _ = stack.shape

        unique_classes, class_counts = np.unique(classification, return_counts=True)
        classes_stats = pd.DataFrame({
            'Classe': unique_classes,
            'Nombre de pixels': class_counts,
            'Pourcentage (%)': (class_counts / np.sum(class_counts) * 100).round(2)
        })
        if 0 in unique_classes:
            valid_pixels_count = np.sum(class_counts) - class_counts[0]
            print(f"Total de pixels classifiés: {valid_pixels_count:,} sur {np.sum(class_counts):,} pixels")
            print(f"Pourcentage classifié: {(valid_pixels_count / np.sum(class_counts) * 100):.2f}%")
            classes_stats_valid = classes_stats[classes_stats['Classe'] > 0].copy()
            classes_stats_valid['Pourcentage valide (%)'] = (classes_stats_valid['Nombre de pixels'] / valid_pixels_count * 100).round(2)
        print("\nStatistiques par classe:")
        print(classes_stats.to_string(index=False))

        plt.figure(figsize=(10, 6))
        classes_to_plot = classes_stats[classes_stats['Classe'] > 0] if 0 in unique_classes else classes_stats
        plt.bar(classes_to_plot['Classe'], classes_to_plot['Nombre de pixels'], color='steelblue')
        plt.xlabel('Classe')
        plt.ylabel('Nombre de pixels')
        plt.title('Distribution des pixels par classe')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(classes_to_plot['Classe'])
        for i, (classe, count, pct) in enumerate(zip(classes_to_plot['Classe'], classes_to_plot['Nombre de pixels'], classes_to_plot['Pourcentage (%)'])):
            plt.text(classe, count + (max(classes_to_plot['Nombre de pixels']) * 0.02), f"{pct}%", ha='center')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, "classes_distribution.png"), dpi=300)
        plt.show()

        try:
            band_names = [src.descriptions[i-1] if i-1 < len(src.descriptions) else f"Bande {i}" for i in SELECTED_BANDS]
            spectral_stats = []
            for class_id in unique_classes:
                if class_id == 0:
                    continue
                class_mask = (classification == class_id)
                for band_idx, band_name in zip(SELECTED_BANDS, band_names):
                    band_data = src.read(band_idx)
                    band_values = band_data[class_mask]
                    spectral_stats.append({
                        'Classe': class_id, 'Bande': band_name,
                        'Moyenne': round(float(np.nanmean(band_values)), 4),
                        'Écart-type': round(float(np.nanstd(band_values)), 4),
                        'Min': round(float(np.nanmin(band_values)), 4),
                        'Max': round(float(np.nanmax(band_values)), 4)
                    })
            spectral_df = pd.DataFrame(spectral_stats)
            print("\nStatistiques spectrales par classe:")
            print(spectral_df.to_string(index=False))
            pivot_means = spectral_df.pivot(index='Classe', columns='Bande', values='Moyenne')
            print("\nMoyennes spectrales par classe:")
            print(pivot_means.to_string())

            plt.figure(figsize=(12, 8))
            for class_id in unique_classes:
                if class_id == 0:
                    continue
                class_data = spectral_df[spectral_df['Classe'] == class_id]
                plt.plot(class_data['Bande'], class_data['Moyenne'], 'o-', label=f"Classe {class_id}")
            plt.title("Signatures spectrales moyennes par classe")
            plt.xlabel("Bande spectrale")
            plt.ylabel("Valeur moyenne")
            plt.grid(True, alpha=0.3)
            plt.legend(loc='best')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "signatures_spectrales.png"), dpi=300)
            plt.show()
        except Exception as e:
            print(f"Impossible de calculer les statistiques spectrales: {e}")

        print("\nCalcul des indices de validation du clustering...")
        try:
            data_for_metrics = stack.reshape(n_bands, -1).T
            valid_pixels = ~np.isnan(data_for_metrics).any(axis=1)
            valid_data = data_for_metrics[valid_pixels]
            valid_labels = classification.flatten()[valid_pixels]
            if 0 in unique_classes:
                valid_indices = valid_labels > 0
                metrics_data = valid_data[valid_indices]
                metrics_labels = valid_labels[valid_indices]
            else:
                metrics_data = valid_data
                metrics_labels = valid_labels

            min_samples_per_class = 10
            unique_labels, label_counts = np.unique(metrics_labels, return_counts=True)
            small_clusters = [label for label, count in zip(unique_labels, label_counts) if count < min_samples_per_class]
            if len(small_clusters) > 0:
                print(f"Attention: {len(small_clusters)} classes ont moins de {min_samples_per_class} pixels.")
                for cluster in small_clusters:
                    print(f"  Classe {cluster}: {label_counts[list(unique_labels).index(cluster)]} pixels")

            calinski_harabasz = metrics.calinski_harabasz_score(metrics_data, metrics_labels)
            print(f"Indice de Calinski-Harabasz: {calinski_harabasz:.4f}")
            print("(Une valeur plus élevée indique une meilleure définition des clusters)")

            davies_bouldin = metrics.davies_bouldin_score(metrics_data, metrics_labels)
            print(f"Indice de Davies-Bouldin: {davies_bouldin:.4f}")
            print("(Une valeur plus proche de zéro indique une meilleure séparation des clusters)")

            max_samples_for_silhouette = 10000
            if metrics_data.shape[0] > max_samples_for_silhouette:
                print(f"Échantillonnage de {max_samples_for_silhouette} pixels sur {metrics_data.shape[0]} pour silhouette...")
                sss = StratifiedShuffleSplit(n_splits=1, test_size=max_samples_for_silhouette, random_state=42)
                _, sample_indices = next(sss.split(metrics_data, metrics_labels))
                silhouette_data = metrics_data[sample_indices]
                silhouette_labels = metrics_labels[sample_indices]
            else:
                silhouette_data = metrics_data
                silhouette_labels = metrics_labels

            unique_labels, counts = np.unique(silhouette_labels, return_counts=True)
            if not np.any(counts < 2):
                silhouette_avg = metrics.silhouette_score(silhouette_data, silhouette_labels)
                print(f"Indice de silhouette moyen: {silhouette_avg:.4f}")
                print("(Une valeur proche de 1 indique une bonne séparation des clusters)")
            else:
                print("Impossible de calculer l'indice de silhouette: certaines classes n'ont qu'un seul élément.")
        except Exception as e:
            print(f"Impossible de calculer les indices de validation: {e}")

        classes_stats.to_csv(os.path.join(OUTPUT_DIR, "classes_statistics.csv"), index=False)
        if 'spectral_df' in locals():
            spectral_df.to_csv(os.path.join(OUTPUT_DIR, "spectral_statistics.csv"), index=False)
        print(f"\nStatistiques exportées dans {OUTPUT_DIR}")
        print("\nAnalyse des statistiques de classification terminée.")