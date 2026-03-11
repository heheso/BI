# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN

# Step 2: Generate synthetic dataset (300 points, 4 centers)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Step 3: Apply K-Means clustering
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)

# Step 4: Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Step 5: Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
# Mark the 'centers' of the clusters with a red X
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, alpha=0.7, marker='X')
plt.title("K-Means Clustering Example")
plt.show()


# eps: maximum distance between two samples for one to be considered as in the neighborhood of the other
# min_samples: number of samples in a neighborhood for a point to be considered a core point
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

# Step 4: Visualize clusters
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='plasma', s=50)
plt.title("DBSCAN Clustering Example")
plt.show()