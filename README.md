# 🔍 KNN Algorithm with PCA and Outlier Detection

This project implements the K-Nearest Neighbors (KNN) algorithm with dimensionality reduction using Principal Component Analysis (PCA) and outlier detection, applied to the Iris dataset. Documentation is in Polish.

---

## ⚙️ Opis działania

1. **Skalowanie danych:** Dane treningowe i testowe są normalizowane za pomocą `StandardScaler`.
2. **Redukcja wymiarowości:** PCA redukuje dane do 2 wymiarów dla klasyfikacji i wizualizacji.
3. **Klasyfikacja KNN:** Klasyfikacja opiera się na k najbliższych sąsiadach (domyślnie k=5) z użyciem metryki euklidesowej.
4. **Wykrywanie outlierów:** Punkt jest oznaczany jako outlier, jeśli większość k najbliższych sąsiadów ma inną etykietę niż prawdziwa.
5. **Wizualizacja:** Wyniki są wyświetlane w przestrzeni 2D, z normalnymi punktami (niebieskie kółka) i outlierami (czerwone krzyżyki).

---

## 📊 Eksperyment

- **Dane:** Zbiór `load_iris()` z `sklearn.datasets`
- **Próbki:** 150
- **Atrybuty:** 4 cechy (długość/szerokość płatka i działki kielicha)
- **Klasy:** 3 gatunki irysa (Setosa, Versicolor, Virginica)

**Metody oceny:**
- Dokładność (accuracy)
- Macierz pomyłek (confusion matrix)
- Wizualizacja outlierów w przestrzeni 2D po PCA
- Rozkład klas i liczba wykrytych outlierów

---

## 🖼️ Wyniki

### 📌 Dokładność

Dokładność modelu KNN na danych testowych wynosi **97.78%**.

### 📌 Macierz pomyłek – KNN

Macierz pomyłek dla klasyfikatora KNN na danych testowych (3 klasy: Setosa, Versicolor, Virginica):

Poniższy wykres słupkowy przedstawia macierz pomyłek klasyfikatora KNN na danych testowych. Oś X oznacza prawdziwe etykiety klas, a każdy kolorowy słupek reprezentuje liczbę próbek przypisanych do danej klasy przez model (przewidywane etykiety):

```json
{
   "type": "bar",
   "data": {
      "labels": ["Setosa", "Versicolor", "Virginica"],
      "datasets": [
         {
            "label": "Przewidywane: Setosa",
            "data": [19, 0, 0],
            "backgroundColor": "#1f77b4"
         },
         {
            "label": "Przewidywane: Versicolor",
            "data": [0, 12, 1],
            "backgroundColor": "#ff7f0e"
         },
         {
            "label": "Przewidywane: Virginica",
            "data": [0, 0, 13],
            "backgroundColor": "#2ca02c"
         }
      ]
   },
   "options": {
      "scales": {
         "x": {
            "title": { "display": true, "text": "Prawdziwe etykiety" }
         },
         "y": {
            "title": { "display": true, "text": "Liczba próbek" },
            "beginAtZero": true
         }
      },
      "plugins": {
         "title": { "display": true, "text": "Macierz pomyłek KNN" },
         "legend": { "display": true }
      }
   }
}
```

### 📌 Rozkład klas w danych testowych

Rozkład klas w zestawie testowym (45 próbek):

```json
{
   "type": "pie",
   "data": {
      "labels": ["Setosa", "Versicolor", "Virginica"],
      "datasets": [{
         "data": [19, 13, 13],
         "backgroundColor": ["#1f77b4", "#ff7f0e", "#2ca02c"],
         "borderColor": ["#ffffff", "#ffffff", "#ffffff"],
         "borderWidth": 1
      }]
   },
   "options": {
      "plugins": {
         "title": { "display": true, "text": "Rozkład klas w danych testowych" },
         "legend": { "display": true }
      }
   }
}
```

### 📌 Liczba wykrytych outlierów

Liczba punktów normalnych (44) i outlierów (1) w zestawie testowym:

```json
{
   "type": "bar",
   "data": {
      "labels": ["Normalne", "Outliery"],
      "datasets": [{
         "label": "Liczba próbek",
         "data": [44, 1],
         "backgroundColor": ["#1f77b4", "#d62728"],
         "borderColor": ["#ffffff", "#ffffff"],
         "borderWidth": 1
      }]
   },
   "options": {
      "scales": {
         "y": {
            "title": { "display": true, "text": "Liczba próbek" },
            "beginAtZero": true
         }
      },
      "plugins": {
         "title": { "display": true, "text": "Liczba normalnych punktów i outlierów" },
         "legend": { "display": false }
      }
   }
}
```

### 📌 Wizualizacja outlierów

Wyniki wykrywania outlierów są wizualizowane w przestrzeni 2D po redukcji PCA, gdzie normalne punkty są oznaczone niebieskimi kółkami, a outliery czerwonymi krzyżykami (zob. `main.py`).

---

## 🧠 Wnioski

- Dokładność 97.78% wskazuje na wysoką skuteczność modelu KNN po redukcji PCA.
- Macierz pomyłek pokazuje pojedynczy błąd w klasyfikacji Versicolor jako Virginica.
- Wykryto tylko jeden outlier, co sugeruje, że dane Iris są spójne.
- Model umożliwia regulację parametru k oraz metryki odległości (euklidesowa lub Manhattan).
- Wizualizacja w 2D ułatwia interpretację wyników i identyfikację anomalii.

---

## 💡 Uruchamianie

```bash
python main.py
```

---

## 📋 Wymagania

- Python 3.x
- Biblioteki: `numpy`, `scikit-learn`, `matplotlib`

---

## 🛠️ Struktura projektu

- `model.py`: Implementacja klasy `KNNCustom` z funkcją klasyfikacji i wykrywania outlierów.
- `main.py`: Skrypt główny, który wczytuje dane Iris, trenuje model, przeprowadza klasyfikację i wizualizuje wyniki.
