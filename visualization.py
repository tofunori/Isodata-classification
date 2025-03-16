import os  # Add this line
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, ListedColormap
import random
from config import OUTPUT_DIR

def visualize_classification(classification):
    """Visualize classification results with a custom colormap."""
    print("Fusion des classes 4 et 7...")
    classification_fusionnee = classification.copy()
    classification_fusionnee[classification == 7] = 4
    print(f"Nombre de pixels modifiés: {np.sum(classification == 7)}")
    classification = classification_fusionnee
    unique_classes = np.unique(classification)
    num_classes = len(unique_classes)
    print(f"Classes après fusion: {unique_classes}")

    plt.figure(figsize=(12, 8))
    hex_colors = ['#FFFFFF00', '#999999', '#00CC33', '#33CC00', '#FF0000', '#0050CC', '#996600', '#38761d', '#FF6699']
    if num_classes > len(hex_colors):
        for i in range(len(hex_colors), num_classes):
            random_color = '#%02X%02X%02X' % (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
            hex_colors.append(random_color)
    rgba_colors = [to_rgba(hex_color) for hex_color in hex_colors[:num_classes]]
    custom_cmap = ListedColormap(rgba_colors)
    valid_classes = [c for c in unique_classes if c > 0]
    print(f"Classes valides (> 0): {valid_classes}")

    class_themes = {1: "Eau", 2: "Forêt", 3: "Végétation herbacée", 4: "Urbain",
                    5: "Sol nu", 6: "Tourbières", 8: "Végétation arbustive", 9: "Milieux humides"}
    class_labels = [f"{class_themes[i]} (4+7)" if i == 4 else f"{class_themes[i]}" if i in class_themes else f"Classe {i}" 
                    for i in valid_classes]

    plt.title("Classification d'occupation du sol", fontsize=14)
    plt.gca().set_facecolor('#EFEFEF')
    im = plt.imshow(classification, cmap=custom_cmap, vmin=0, vmax=num_classes-1)
    cax = plt.axes([1, 0.15, 0.02, 0.7])
    cbar = plt.colorbar(im, cax=cax, ticks=valid_classes)
    cbar.set_label("Types d'occupation du sol", fontsize=10)
    cbar.ax.set_yticklabels(class_labels, fontsize=8)
    plt.axis('on')
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "classification_sans_classe0.png"), dpi=300, transparent=False)
    plt.show()
    print("\nCarte de classification sans classe 0 (noire) générée avec succès!")

if __name__ == "__main__":
    # For testing purposes, you can add a dummy classification array
    import numpy as np
    dummy_classification = np.random.randint(0, 10, size=(100, 100))
    visualize_classification(dummy_classification)