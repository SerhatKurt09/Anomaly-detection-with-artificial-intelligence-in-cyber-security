# train_model.py

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import joblib
import matplotlib.pyplot as plt

# Veriyi oku
df = pd.read_csv("Data/Cleaned/friday_cleaned.csv")
y = df["Label"]
X = df.drop(columns=["Label"])

# Özellik sırasını koru
with open("model/features.txt", "r") as f:
    feature_list = [line.strip() for line in f.readlines()]
X = X[feature_list]

# Ölçekleme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim/test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)

# Model
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Test performansı
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"✅ Test Doğruluğu: {acc:.4f}")
print(f"✅ Test F1 Skoru: {f1:.4f}")
print("\n🔍 Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Saldırı"]))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 5-kat çapraz doğrulama
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
print(f"\n🧪 5-Kat CV F1 Skorları: {cv_scores}")
print(f"📈 Ortalama CV F1 Skoru: {cv_scores.mean():.4f}")

# Grafik: CV skorları
plt.figure(figsize=(8, 5))
plt.plot(range(1, 6), cv_scores, marker='o', linestyle='--', color='b')
plt.title("5-Kat Çapraz Doğrulama F1 Skorları")
plt.xlabel("Kat (Fold)")
plt.ylabel("F1 Skoru (Macro)")
plt.ylim(0, 1)
plt.grid(True)
plt.tight_layout()
plt.show()

# Grafik: Öğrenme eğrisi
train_sizes, train_scores, valid_scores = learning_curve(
    model, X_train, y_train, cv=5, scoring='f1_macro', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10), random_state=42
)
train_scores_mean = np.mean(train_scores, axis=1)
valid_scores_mean = np.mean(valid_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
valid_scores_std = np.std(valid_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', label="Eğitim Skoru", color='blue')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='blue')

plt.plot(train_sizes, valid_scores_mean, 'o-', label="Doğrulama Skoru", color='green')
plt.fill_between(train_sizes, valid_scores_mean - valid_scores_std,
                 valid_scores_mean + valid_scores_std, alpha=0.1, color='green')

plt.title("Öğrenme Eğrisi (F1 Macro Skoru)")
plt.xlabel("Eğitim Seti Boyutu")
plt.ylabel("F1 Skoru")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# Model ve scaler kaydet
joblib.dump(model, "model/random_forest_model.pkl")
joblib.dump(scaler, "model/scaler.pkl")
print("\n💾 Model ve scaler kaydedildi: model/random_forest_model.pkl & scaler.pkl")
