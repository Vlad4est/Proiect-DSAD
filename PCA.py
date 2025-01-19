import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("HouseData.csv")

# 2. Calculăm vârsta clădirilor
current_year = 2025
#data["Age"] = current_year - data["YearBuilt"]


import seaborn as sns
import matplotlib.pyplot as plt

# Matricea de corelație
corr_matrix = data[["Sq.Ft", "Beds", "Bath", "Price"]].corr()

# Vizualizarea matricei de corelație
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Matricea de Corelație")
plt.show()
# 3. Selectăm variabilele numerice pentru PCA
numerical_features = ["Sq.Ft", "Beds", "Bath", "Price"]
data_pca = data[numerical_features]

# 4. Standardizăm variabilele
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_pca)

# 5. Aplicăm PCA
pca = PCA()
pca_components = pca.fit_transform(data_scaled)

# 6. Calculăm varianța explicată
total_variance = np.cumsum(pca.explained_variance_ratio_)
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)
# Rezultate intermediare
print("Proporția variației explicate de fiecare component principal:", pca.explained_variance_ratio_)
print("Variația cumulativă explicată:", total_variance)

# 7. Grafic pentru variația explicată
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(total_variance) + 1), total_variance, marker='o', linestyle='--')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()

# 8. Interpretarea primilor 2 componenti
pca_df = pd.DataFrame(pca_components, columns=[f"PC{i+1}" for i in range(len(numerical_features))])
pca_df["PC1"].head(), pca_df["PC2"].head()

# Contribuția variabilelor la fiecare component principal
pca_loadings = pd.DataFrame(pca.components_, columns=numerical_features, index=[f"PC{i+1}" for i in range(len(numerical_features))])
print("Contribuția variabilelor la componentele principale:")
print(pca_loadings)


# Salvăm rezultatele (dacă este necesar)
pca_df.to_csv("pca_results.csv", index=False)

# 1. Biplot clarificat
plt.figure(figsize=(12, 10))
for i, var in enumerate(data.columns):
    plt.arrow(0, 0, pca.components_[0, i], pca.components_[1, i],
              color='red', alpha=0.8, head_width=0.05, head_length=0.05)
    plt.text(pca.components_[0, i] * 1.2, pca.components_[1, i] * 1.2, var,
             color='green', ha='center', va='center', fontsize=12)

# Adăugăm scatter plot-ul cu observații
plt.scatter(pca_components[:, 0], pca_components[:, 1],
            alpha=0.4, s=10, c='blue', edgecolor=None)

# Titlu și elemente estetice
plt.title('Biplot of PCA (PC1 vs PC2)', fontsize=14)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.grid(alpha=0.5)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
plt.show()
# 2. Cumulative Explained Variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', label='Cumulative Variance')
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.6, label='Individual Variance')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Variance Explained')
plt.xticks(range(1, len(explained_variance) + 1))
plt.legend()
plt.grid()
plt.show()

# 3. Scatter plot of PC1 vs PC2
# 2. Scatter plot clarificat
plt.figure(figsize=(12, 10))

# Folosim o hartă de culori pentru densitate
scatter = plt.scatter(pca_components[:, 0], pca_components[:, 1],
                       alpha=0.4, s=10, c=pca_components[:, 0],
                       cmap='viridis', edgecolor=None)

# Adăugăm o bară de culoare pentru densitate
plt.colorbar(scatter, label='Density (PC1)', shrink=0.8)

# Titlu și elemente estetice
plt.title('Projection of Data onto PC1 and PC2', fontsize=14)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)
plt.grid(alpha=0.5)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
plt.axvline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.7)
plt.show()


# 4. Heatmap of PCA Loadings
plt.figure(figsize=(10, 6))
sns.heatmap(pca_loadings, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
plt.title('PCA Loadings Heatmap')
plt.show()
