#Random Forest Sınıflandırma
# Gerekli kütüphaneleri içe aktar
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Olivetti yüz veri setini yükle
oli = fetch_olivetti_faces()

# İlk iki yüz görüntüsünü görselleştir
plt.figure()
for i in range(2):
    plt.subplot(1, 2, i + 1)
    plt.imshow(oli.images[i], cmap='gray')  # Görüntüyü gri tonlarda göster
    #plt.imshow(oli.images[i + 40], cmap='gray')  # Alternatif olarak başka bir yüz
    plt.axis("off")  # Eksenleri gizle
plt.show()

# Özellikler (X) ve hedef (y) değişkenlerini ayır
X = oli.data
y = oli.target

# Veriyi eğitim ve test olarak ayır (test %20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Farklı ağaç sayılarıyla dört Random Forest modeli oluştur
rf_clf1 = RandomForestClassifier(n_estimators=5, random_state=42)
rf_clf2 = RandomForestClassifier(n_estimators=20, random_state=42)
rf_clf3 = RandomForestClassifier(n_estimators=50, random_state=42)
rf_clf4 = RandomForestClassifier(n_estimators=100, random_state=42)

# Modelleri eğit
rf_clf1.fit(X_train, y_train)
rf_clf2.fit(X_train, y_train)
rf_clf3.fit(X_train, y_train)
rf_clf4.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred1 = rf_clf1.predict(X_test)
y_pred2 = rf_clf2.predict(X_test)
y_pred3 = rf_clf3.predict(X_test)
y_pred4 = rf_clf4.predict(X_test)

# Doğruluk skorlarını hesapla
accuracy1 = accuracy_score(y_test, y_pred1)
accuracy2 = accuracy_score(y_test, y_pred2)
accuracy3 = accuracy_score(y_test, y_pred3)
accuracy4 = accuracy_score(y_test, y_pred4)

# Sonuçları ekrana yazdır
print("Doğruluk1 : ", accuracy1)  
print("Doğruluk2 : ", accuracy2)  
print("Doğruluk3 : ", accuracy3)  
print("Doğruluk4 : ", accuracy4)

