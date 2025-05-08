# Since we don't have actual interactions between users and songs,
# we simulate them by assigning each track a 'percentage_listened' value and a 'liked' flag (1 for liked, 0 for not liked).
# This simulation is necessary because real-time interaction data is not available for training.
# Therefore, we opt for a simulated approach.
# For each cluster, we assign 'percentage_listened' values following a Gaussian distribution (0 to 1).

import pandas as pd
import numpy as np

df = pd.read_csv("cluster_songs.csv")

# Assigning per-cluster uniform parameters (mean, std)
cluster_ids = df['cluster'].unique()
cluster_stats = {
    cluster: {
        'mean': np.random.uniform(0.4, 0.9),
        'std': np.random.uniform(0.05, 0.15)
    }
    for cluster in cluster_ids}

# Simulate percentage_listened using Gaussian sampling
def sample_percentage(cluster_id):
    mean = cluster_stats[cluster_id]['mean']
    std = cluster_stats[cluster_id]['std']
    val = np.random.normal(loc=mean, scale=std)
    return float(np.clip(val, 0.0, 1.0))  # value should range between 0-1
df['percentage_listened'] = df['cluster'].apply(sample_percentage)
# If percentage_listened > 0.6 consider user liked the track
df['liked'] = df['percentage_listened'].apply(lambda x: int(x > 0.6))
df.to_csv('simulated_songs_interactions.csv', index=False)
