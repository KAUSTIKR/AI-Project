# The simulated 'percentage_listened' and 'liked' values are used to compute the interaction score.
# A proximity score is also calculated — tracks closer to the cluster centroid receive higher scores,
# while those farther away receive lower scores.
# By combining the interaction score and the proximity score, we calculate the final reward.
# This reward is higher for tracks near the centroid and gradually decreases for tracks further away.
# The final reward is assigned to each track and serves as the reward signal for being in that state during PPO training).
# The output of this program serves as the input dataset for training the PPO model.

import pandas as pd
import numpy as np

df = pd.read_csv('simulated_songs_interactions.csv')
latent_cols = ['latent_0', 'latent_1', 'latent_2', 'latent_3', 'latent_4']
centroids = df.groupby('cluster')[latent_cols].mean()

# Calculate distance to clusters centroid
def compute_distance(row):
    center = centroids.loc[row['cluster']].values
    song_vec = row[latent_cols].values
    return np.linalg.norm(song_vec - center)

df['distance_to_centroid'] = df.apply(compute_distance, axis=1)

# Normalizing distance within cluster for proximity score
df['max_distance_in_cluster'] = df.groupby('cluster')['distance_to_centroid'].transform('max')
df['proximity_score'] = 1 - (df['distance_to_centroid'] / df['max_distance_in_cluster'])

# Calculate interaction score
w1, w2 = 100.0, 50  # weights
df['interaction_score'] = w1 * df['percentage_listened'] + w2 * df['liked']

alpha, beta = 0.8, 0.2  # weights for final reward
df['reward'] = alpha * df['interaction_score'] + beta * df['proximity_score']
df.to_csv('ppo_input.csv', index=False)
df[['id', 'latent_0', 'latent_1', 'latent_2', 'latent_3', 'latent_4', 'cluster', 'percentage_listened', 'liked', 'interaction_score', 'proximity_score', 'reward']] \
  .to_json('combined_reward_songs.json', orient='records', indent=2)
