# Gerekli kütüphaneleri içe aktar
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Diabetes veri setini yükle
diabets = load_diabetes()
X = diabets.data
y = diabets.target

# Veriyi eğitim ve test olarak ayır (test %20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Karar ağacı regresyon modelini oluştur ve eğit
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = tree_reg.predict(X_test)

# Ortalama kare hata (MSE) hesapla
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Kök ortalama kare hata (RMSE) hesapla
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

