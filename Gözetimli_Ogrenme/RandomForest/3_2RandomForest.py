# Random Forest Regresyon

# Gerekli kütüphaneleri içe aktar
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# California konut veri setini yükle
california_housing = fetch_california_housing()

# Özellikler (X) ve hedef (y) değişkenlerini ayır
X = california_housing.data
y = california_housing.target

# Veriyi eğitim ve test olarak ayır (test %20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest regresyon modelini oluştur ve eğit
rf_reg = RandomForestRegressor(random_state=42)
rf_reg.fit(X_train, y_train)

# Test verisi ile tahmin yap
y_pred = rf_reg.predict(X_test)

# Ortalama kare hata (MSE) hesapla
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Kök ortalama kare hata (RMSE) hesapla
rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)
