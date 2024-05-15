import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn_extra.cluster import KMedoids

data = pd.read_excel("climate.xlsx")
X = data.iloc[:, 1:].values
np.random.seed(42)

n_clusters = 4
eps = 10  # 邻域半径
min_samples = 5  # 最小样本数
m = 2.0
error = 0.005
max_iter = 1000

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan.fit(X)
labels_dbscan = dbscan.labels_
n_clusters_ = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
clusters_dbscan = [X[labels_dbscan == i] for i in range(n_clusters_)]

sse = np.sum([np.sum((cluster - np.mean(cluster, axis=0)) ** 2) for cluster in clusters_dbscan])
silhouette_avg = silhouette_score(X, labels_dbscan)
print("DBSCAN:")
print("SSE:", sse)
print("Silhouette Score:", silhouette_avg)

kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
clusters_kmeans = kmeans.fit_predict(X)
labels_kmeans = kmeans.labels_

sse = kmeans.inertia_
silhouette_avg = silhouette_score(X, clusters_kmeans)
print("k-means++:")
print("SSE:", sse)
print("Silhouette Score:", silhouette_avg)

spectral_clustering = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42)
clusters_sc = spectral_clustering.fit_predict(X)
labels_sc = spectral_clustering.labels_
cluster_centers = []

for i in range(n_clusters):
    cluster_center = np.mean(X[clusters_sc == i], axis=0)
    cluster_centers.append(cluster_center)

cluster_centers = np.array(cluster_centers)

sse = 0
for i in range(len(X)):
    cluster_center = cluster_centers[clusters_sc[i]]
    sse += np.sum((X[i] - cluster_center) ** 2)

silhouette_avg = silhouette_score(X, clusters_sc)
print("spectral clustering:")
print("SSE:", sse)
print("Silhouette Score:", silhouette_avg)

# 初始化FCM
cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(X.T, n_clusters, m, error, max_iter)

# 计算模糊分配矩阵中每个样本所属的聚类
cluster_fcm = np.argmax(u, axis=0)

sse = 0
for i in range(len(X)):
    cluster_center = cntr[cluster_fcm[i]]
    sse += np.sum((X[i] - cluster_center) ** 2)

silhouette_avg = silhouette_score(X, cluster_fcm)
print("FCM:")
print("SSE:", sse)
print("Silhouette Score:", silhouette_avg)

ac = AgglomerativeClustering(n_clusters=n_clusters)
cluster_ac = ac.fit(X)
labels_ac = ac.labels_

clusters_ac = [X[labels_ac == i] for i in range(n_clusters)]
sse = np.sum([np.sum((cluster - np.mean(cluster, axis=0)) ** 2) for cluster in clusters_ac])
silhouette_avg = silhouette_score(X, labels_ac)
print("Agglomerative Clustering:")
print("SSE:", sse)
print("Silhouette Score:", silhouette_avg)

kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
cluster_km = kmedoids.fit(X)
labels_km = cluster_km.labels_

clusters_dbscan = [X[labels_km == i] for i in range(n_clusters)]
sse = np.sum([np.sum((cluster - np.mean(cluster, axis=0)) ** 2) for cluster in clusters_dbscan])
silhouette_avg = silhouette_score(X, labels_ac)
print("K-medoids Clustering:")
print("SSE:", sse)
print("Silhouette Score:", silhouette_avg)
# 使用PCA将多维数据降至二维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# 绘制降维后的聚类结果散点图
plt.figure(figsize=(15, 8))

# 绘制DBSCAN的结果
plt.subplot(231)
noise = X_pca[labels_dbscan == -1]
plt.scatter(noise[:, 0], noise[:, 1], color='gray', label='Noise')
for i in range(n_clusters_):
    cluster = X_pca[labels_dbscan == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')

plt.title('DBSCAN Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# 绘制KMeans的结果
plt.subplot(232)
for i in range(n_clusters):
    cluster = X_pca[clusters_kmeans == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')
plt.title('KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# 绘制Spectral Clustering的结果
plt.subplot(233)
for i in range(n_clusters):
    cluster = X_pca[clusters_sc == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')
plt.title('Spectral Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# 绘制FCM的结果
plt.subplot(234)
for i in range(n_clusters):
    cluster = X_pca[cluster_fcm == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')
plt.title('FCM Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# 绘制Agglomerative Clustering的结果
plt.subplot(235)
for i in range(n_clusters):
    cluster = X_pca[labels_ac == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')
plt.title('Agglomerative Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.subplot(236)
for i in range(n_clusters):
    cluster = X_pca[labels_km == i]
    plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i + 1}')
plt.title('KMedoids Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

plt.tight_layout()
plt.show()
