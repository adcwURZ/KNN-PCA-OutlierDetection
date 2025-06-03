import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from model import KNNCustom

# Wczytanie danych Iris
data = load_iris()
X = data.data
y = data.target
class_names = data.target_names  # ['setosa', 'versicolor', 'virginica']

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Redukcja wymiarowości za pomocą PCA
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Trenowanie modelu KNN
model = KNNCustom(k=5, distance_m='euclid')
model.fit(X_train_pca, y_train)

# Predykcja i ocena
y_pred = model.predict(X_test_pca)
acc = accuracy_score(y_test, y_pred)
conf = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion matrix:\n", conf)

# Wykrywanie outlierów
outliers = model.predict(X_test_pca, y_true=y_test, detect_outliers=True)
print("Outlier labels:", outliers)
print("Number of detected outliers:", np.sum(outliers))

# Wizualizacja 1: Outliery w przestrzeni 2D po PCA
plt.figure(figsize=(8, 6))
normal_indices = np.where(outliers == 0)[0]
outlier_indices = np.where(outliers == 1)[0]

plt.scatter(X_test_pca[normal_indices, 0], X_test_pca[normal_indices, 1],
            c='blue', label='Normalne', marker='o', alpha=0.6, s=100)
plt.scatter(X_test_pca[outlier_indices, 0], X_test_pca[outlier_indices, 1],
            c='red', label='Outliery', marker='x', alpha=0.8, s=150, linewidths=2)

plt.title("Wizualizacja Outlierów z użyciem PCA", fontsize=14, pad=15)
plt.xlabel("PCA1", fontsize=12)
plt.ylabel("PCA2", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Wizualizacja 2: Macierz pomyłek jako heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title("Macierz Pomyłek KNN", fontsize=14, pad=15)
plt.xlabel("Przewidywane etykiety", fontsize=12)
plt.ylabel("Prawdziwe etykiety", fontsize=12)
plt.tight_layout()
plt.show()

# Wizualizacja 3: Rozkład klas w danych testowych (wykres kołowy)
class_counts = [np.sum(y_test == i) for i in range(3)]  # [19, 13, 13]
plt.figure(figsize=(8, 6))
plt.pie(class_counts, labels=class_names, autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e', '#2ca02c'],
        startangle=90, textprops={'fontsize': 12})
plt.title("Rozkład klas w danych testowych", fontsize=14, pad=15)
plt.tight_layout()
plt.show()