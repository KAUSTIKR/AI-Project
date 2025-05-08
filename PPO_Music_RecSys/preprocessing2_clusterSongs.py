# In our recommendation problem we don't user-song interaction details.
# Which is a cold start problem.
# Cold start problem can be solved using content based filter or Kmeans clustering.
# Clustering helps group similar tracks into the same cluster.
# This provides a meaningful way to identify related songs and offers a strong starting point for recommendations.

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from numpy.linalg import norm

input_path = 'vae_latent_vectors.csv'
df = pd.read_csv(input_path)

latent_cols = [col for col in df.columns if 'latent' in col]
X = df[latent_cols].values

# Applying Kmeans
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X)

# Get Top 4 songs closest to each cluster centroid
top_4_list = []
for cluster_id in range(n_clusters):
    cluster_songs = df[df['cluster'] == cluster_id]
    if cluster_songs.empty:
        continue
    # Compute distance to cluster centroid
    centroid = kmeans.cluster_centers_[cluster_id]
    print(centroid)
    print(cluster_id)
    vectors = cluster_songs[latent_cols].values
    distances = norm(vectors - centroid, axis=1)
    # Take only top 4
    cluster_songs = cluster_songs.copy()
    cluster_songs['distance_to_centroid'] = distances
    top_4 = cluster_songs.nsmallest(4, 'distance_to_centroid')
    top_4_list.append(top_4)

top_4_df = pd.concat(top_4_list, ignore_index=True)
df.to_csv('cluster_songs.csv', index=False)
# top_4_df.to_csv('top4_songs_per_cluster.csv', index=False)
print("Files saved: cluster_songs.csv")
