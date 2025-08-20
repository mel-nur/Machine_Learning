# Gerekli kütüphaneleri içe aktar
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Iris veri setini yükle
iris = load_iris()

# Özellikler (X) ve hedef (y) değişkenlerini ayır
X = iris.data
y = iris.target

# Veriyi eğitim ve test olarak ayır (test %20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Naive Bayes sınıflandırıcı modelini oluştur ve eğit
nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = nb_clf.predict(X_test)

# Sınıflandırma raporunu ekrana yazdır
print(classification_report(y_test, y_pred))
