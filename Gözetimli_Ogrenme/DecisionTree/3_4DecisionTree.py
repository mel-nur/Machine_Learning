# Gerekli kütüphaneleri içe aktar
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

# Rasgele veri seti oluştur
X = np.sort(5 * np.random.rand(80, 1), axis=0)  # 0-5 arası 80 örnek
y = np.sin(X).ravel()  # Sinüs fonksiyonu ile hedef değerler
# Her 5. örneğe gürültü ekle
y[::5] += 0.5 * (0.5 - np.random.rand(16))

# İki farklı derinlikte karar ağacı regresyon modeli oluştur
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
# Modelleri veriye fit et
regr_1.fit(X, y)
regr_2.fit(X, y)

# Test verisi oluştur
X_test = np.arange(0, 5, 0.05)[:, np.newaxis]
# Test verisi ile tahmin yap
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

# Sonuçları görselleştir
plt.figure()
# Orijinal veriyi çiz
plt.plot(X, y, c="red", label="data")
plt.scatter(X, y, c="red", label="data")
# Farklı derinlikteki modellerin tahminlerini çiz
plt.plot(X_test, y_pred_1, color="blue", label="Max Depth: 2", linewidth=2)
plt.plot(X_test, y_pred_2, color="green", label="Max Depth: 5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()
plt.show()
         