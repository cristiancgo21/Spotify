# =======================
# 1. CARGA Y PREPROCESO
# =======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# --- Cargar dataset ---
df = pd.read_csv("spotify_churn_dataset.csv")

# Seleccionar columnas numéricas (ajusta a tu caso)
num_cols = df.select_dtypes(include=[np.number]).columns
X = df[num_cols]

# Escalar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =======================
# 2. K-MEANS
# =======================
# Inicial con K=3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)

# Evaluar k óptimo con codo y silueta
inertia = []
silhouette = []
K = range(2, 10)

for k in K:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X_scaled, labels))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K, inertia, 'o-')
plt.title('Método del Codo')
plt.xlabel('Número de clusters')
plt.ylabel('Inercia')

plt.subplot(1,2,2)
plt.plot(K, silhouette, 'o-')
plt.title('Coeficiente Silueta')
plt.xlabel('Número de clusters')
plt.ylabel('Silhouette')
plt.tight_layout()
plt.show()

# =======================
# 3. CLUSTERING JERÁRQUICO
# =======================
# Dendrograma para decidir número de clusters
linked = linkage(X_scaled, method='ward')
plt.figure(figsize=(10,6))
dendrogram(linked)
plt.title('Dendrograma - Clustering Jerárquico')
plt.xlabel('Observaciones')
plt.ylabel('Distancia')
plt.show()

# Por ejemplo, cortar en 3 clusters:
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
df['Cluster_HC'] = hc.fit_predict(X_scaled)

# =======================
# 4. DBSCAN
# =======================
# Probar con eps y min_samples (ajustar valores)
dbscan = DBSCAN(eps=1.5, min_samples=5)
df['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)

# =======================
# 5. VISUALIZACIÓN PCA
# =======================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(16,4))

plt.subplot(1,3,1)
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster_KMeans'], cmap='viridis', alpha=0.6)
plt.title('K-Means')
plt.xlabel('PC1'); plt.ylabel('PC2')

plt.subplot(1,3,2)
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster_HC'], cmap='plasma', alpha=0.6)
plt.title('Clustering Jerárquico')
plt.xlabel('PC1'); plt.ylabel('PC2')

plt.subplot(1,3,3)
plt.scatter(X_pca[:,0], X_pca[:,1], c=df['Cluster_DBSCAN'], cmap='coolwarm', alpha=0.6)
plt.title('DBSCAN')
plt.xlabel('PC1'); plt.ylabel('PC2')

plt.tight_layout()
plt.show()

# =======================
# 6. RESUMEN
# =======================
print("\nCantidad de elementos por cluster K-Means:")
print(df['Cluster_KMeans'].value_counts())

print("\nCantidad de elementos por cluster Jerárquico:")
print(df['Cluster_HC'].value_counts())

print("\nCantidad de elementos por cluster DBSCAN (-1 = ruido):")
print(df['Cluster_DBSCAN'].value_counts())

# ===========================
# 7. TABLA COMPARATIVA MÉTRICAS
# ===========================
from sklearn.metrics import silhouette_score
import pandas as pd

metrics = []

# --- KMeans ---
# (ejemplo con 3 clusters, pero puedes iterar varios k)
labels_kmeans = df['Cluster_KMeans']
sil_kmeans = silhouette_score(X_scaled, labels_kmeans)
metrics.append({
    'Método': 'K-Means',
    'Parámetros': 'n_clusters=3',
    'Silhouette Score': sil_kmeans
})

# --- Clustering Jerárquico ---
labels_hc = df['Cluster_HC']
sil_hc = silhouette_score(X_scaled, labels_hc)
metrics.append({
    'Método': 'Jerárquico',
    'Parámetros': 'n_clusters=3, linkage=ward',
    'Silhouette Score': sil_hc
})

# --- DBSCAN ---
labels_dbscan = df['Cluster_DBSCAN']
# DBSCAN puede dejar puntos como ruido (-1), solo calcula silhouette si hay >=2 clusters distintos
if len(set(labels_dbscan)) > 1 and -1 not in set(labels_dbscan):
    sil_dbscan = silhouette_score(X_scaled, labels_dbscan)
else:
    sil_dbscan = 'No aplicable (ruido o 1 cluster)'
metrics.append({
    'Método': 'DBSCAN',
    'Parámetros': 'eps=1.5, min_samples=5',
    'Silhouette Score': sil_dbscan
})

# Crear dataframe comparativo
df_metrics = pd.DataFrame(metrics)
print("\n=== Comparativa de Silhouette Score ===")
print(df_metrics)