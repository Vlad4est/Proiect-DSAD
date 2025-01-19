import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns


def load_and_prepare_data(file_path):
        """Încarcă și pregătește datele pentru analiză"""
        # Citim datele din CSV
        data = pd.read_csv("HouseData.csv")

        # Selectăm coloanele pentru analiză
        features = ['Price', 'Beds', 'Bath', 'Sq.Ft']
        X = data[features]

        # Standardizăm datele
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=features)

        return X, X_scaled, data


def plot_dendrogram(X_scaled):
        """Creează și afișează dendrograma"""
        plt.figure(figsize=(10, 7))
        linkage_matrix = linkage(X_scaled, method='ward')
        dendrogram(linkage_matrix)
        plt.title('Dendrograma Analizei de Cluster Ierarhic')
        plt.xlabel('Index Sample')
        plt.ylabel('Distanță')
        plt.show()


def perform_clustering(X_scaled, n_clusters=3):
        """Realizează clusterizarea și returnează rezultatele"""
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        clusters = clustering.fit_predict(X_scaled)
        return clusters


def analyze_clusters(data, clusters):
        """Analizează caracteristicile fiecărui cluster"""
        data['Cluster'] = clusters
        cluster_stats = data.groupby('Cluster').agg({
                'Price': ['mean', 'min', 'max', 'count'],
                'Beds': 'mean',
                'Bath': 'mean',
                'Sq.Ft': 'mean'
        }).round(2)

        return cluster_stats


def plot_cluster_visualization(X_scaled, clusters):
        """Creează vizualizări pentru clustere folosind primele două caracteristici"""
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(X_scaled.iloc[:, 0], X_scaled.iloc[:, 1],
                              c=clusters, cmap='viridis')
        plt.xlabel('Preț (standardizat)')
        plt.ylabel('Dormitoare (standardizat)')
        plt.title('Vizualizare Clustere')
        plt.colorbar(scatter)
        plt.show()


# Specificați calea către fișierul CSV
file_path = '"HouseData.csv"'  # Înlocuiți cu calea reală către fișierul dvs.

# Executăm analiza
X, X_scaled, data = load_and_prepare_data(file_path)

# Afișăm dendrograma
plot_dendrogram(X_scaled)

# Realizăm clusterizarea
clusters = perform_clustering(X_scaled, n_clusters=3)

# Analizăm rezultatele
cluster_stats = analyze_clusters(data, clusters)
print("\nStatistici pentru fiecare cluster:")
print(cluster_stats)

# Vizualizăm clusterele
plot_cluster_visualization(X_scaled, clusters)

# Salvăm rezultatele într-un CSV nou
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = clusters
data_with_clusters.to_csv('rezultate_clustering.csv', index=False)
