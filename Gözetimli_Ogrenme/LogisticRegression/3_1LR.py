# pip install ucimlrepo
# Logistic Regression
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Uyarıları gizle
import warnings
warnings.filterwarnings("ignore")

# Heart Disease veri setini UCI'dan çek
heart_disease = fetch_ucirepo(id=45)

# Özellikler ve hedefi DataFrame'e aktar
df = pd.DataFrame(heart_disease.data.features)
df["target"] = heart_disease.data.targets

# Eksik değerleri kontrol et ve varsa sil
if df.isna().any().any():  # Eksik değer var mı?
    df.dropna(inplace=True)  # Eksik satırları sil
    print("nan")

# Özellikler (X) ve hedef (y) değişkenlerini ayır
X = df.drop(["target"], axis=1).values
y = df.target.values

# Veriyi eğitim ve test olarak ayır (test %10)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Lojistik regresyon modelini oluştur ve eğit
log_reg = LogisticRegression(penalty='l2', C=1, solver='lbfgs', random_state=42)
log_reg.fit(X_train, y_train)

# Test verisi ile doğruluk skorunu hesapla
accuracy = log_reg.score(X_test, y_test)
print("Logistic Regression Accuracy : " , accuracy)

# Ek veri ve metadata örnekleri (yorum satırı)
"""
# data (as pandas dataframes)
# X = heart_disease.data.features
# y = heart_disease.data.targets
# metadata
# print(heart_disease.metadata)
# variable information
# print(heart_disease.variables)
"""