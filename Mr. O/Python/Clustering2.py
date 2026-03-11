#==========================================
# Clustering using DBSCAN with make_blobs
#==========================================
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

#================
# Generate Dataset
#================
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.50, random_state=0)

#======================
# Apply DBSCAN Algorithm
#======================
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

#==================
# Plot the Clusters
#==================
plt.scatter(X[:,0], X[:,1], c=labels, cmap='plasma', s=50)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

#=============
# Final Answer
#=============
# DBSCAN groups the dataset into clusters based on density and
# identifies noise points automatically.