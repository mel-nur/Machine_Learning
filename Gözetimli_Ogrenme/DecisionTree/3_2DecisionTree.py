# Gözetimli Öğrenme: Decision Tree (Karar Ağaçları)
# Gerekli kütüphaneleri içe aktar
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
import warnings

# Uyarıları gizle (örneğin, colormap ile ilgili)
warnings.filterwarnings("ignore")

# Iris veri setini yükle
iris = load_iris()

# Sınıf sayısını ve renkleri tanımla
n_classes = len(iris.target_names)
plot_colors = "ryb"

# Özellik çiftleri üzerinden döngü
for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]]):
    # Seçilen iki özellik ile X ve hedef y'yi oluştur
    X = iris.data[:, pair]
    y = iris.target
    # Karar ağacı modelini eğit
    clf = DecisionTreeClassifier().fit(X, y)
    # Her özellik çifti için bir subplot oluştur
    ax = plt.subplot(2, 3, pairidx + 1)
    # Grafik düzenini ayarla
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)
    # Karar sınırlarını çiz
    DecisionBoundaryDisplay.from_estimator(
        clf, X,
        cmap=plt.cm.RdYlBu,
        response_method="predict",
        ax=ax,
        xlabel=iris.feature_names[pair[0]],
        ylabel=iris.feature_names[pair[1]]
    )
    # Her sınıf için veri noktalarını scatter ile göster
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], c=color, label=iris.target_names[i], edgecolor='black', s=20)
    # Legend ekle
    ax.legend()

# Tüm grafiklerin ekranda gösterilmesi
plt.show()