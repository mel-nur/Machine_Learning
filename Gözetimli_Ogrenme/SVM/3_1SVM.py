# Support Vector Machine (Destek Vektör Makinesi)
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Rakam veri setini yükle
digits = load_digits()

# İlk 10 rakam görüntüsünü görselleştir
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5),
            subplot_kw={'xticks': [], 'yticks': []})

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap="binary", interpolation="nearest")  # Görüntüyü siyah-beyaz göster
    ax.set_title(digits.target[i])  # Görüntünün rakam etiketini başlığa yaz
plt.show()

# Özellikler (X) ve hedef (y) değişkenlerini ayır
X = digits.data
y = digits.target

# Veriyi eğitim ve test olarak ayır (test %20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM modelini oluştur ve eğit
svm_clf = SVC(kernel="linear", random_state=42)
svm_clf.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = svm_clf.predict(X_test)

# Sınıflandırma raporunu ekrana yazdır
print(classification_report(y_test, y_pred))
