#============================================
# Clustering using K-Means with make_blobs
#============================================
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

#=======================
# Generate Sample Dataset
#=======================
X, y = make_blobs(n_samples=300, centers=4, random_state=0)

#====================
# Apply K-Means Clustering
#====================
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

#==================
# Plot the Clusters
#==================
plt.scatter(X[:,0], X[:,1], c=labels, s=50, cmap='viridis')
plt.scatter(centroids[:,0], centroids[:,1], s=200, alpha=0.7, marker='X')
plt.title("K-Means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#============
# Final Answer
#============
# The dataset generated using make_blobs is grouped into 4 clusters
# using the K-Means clustering algorithm.
