#Gözetimli Öğrenme Decision Tree(Karar Ağaçları)
# Gerekli kütüphaneleri içe aktar
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Iris veri setini yükle
iris = load_iris()

# Özellikler (feature) ve hedef (target) değişkenlerini ayır
X = iris.data  # Özellikler
y = iris.target  # Hedef değişken

# Veriyi eğitim ve test olarak ayır (test %20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı (Decision Tree) modelini oluştur ve eğit
tree_clf = DecisionTreeClassifier(criterion="gini", max_depth=5, random_state=42)
tree_clf.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = tree_clf.predict(X_test)

# Doğruluk oranını hesapla
accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk : ", accuracy)

# Karışıklık matrisi (confusion matrix) hesapla
confusion_mat = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_mat)

# Karar ağacını görselleştir
plt.figure(figsize=(15, 10))
plot_tree(tree_clf, filled=True, feature_names=iris.feature_names, class_names=list(iris.target_names))
plt.show()

# Özelliklerin önem derecelerini al
feature_importances = tree_clf.feature_importances_
feature_names = iris.feature_names
feature_importances_sorted = sorted(zip(feature_importances, feature_names), reverse=True)

# Özellik önemlerini ekrana yazdır
for importance, feature_name in feature_importances_sorted:
    print(f"{feature_name} : {importance}")



